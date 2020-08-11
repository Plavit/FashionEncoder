import tensorflow as tf

_NEG_INF_FP32 = -1e9
_INF_FP32 = 1e9


def _get_mask_positions_weights(mask_positions, categories):
    """
    Get weights of shape [batch_size * seq_length] with values 1 and 0
        - 1 for positions of mask tokens
        - 0 for the rest
    Args:
        mask_positions: int tensor with shape [batch_size, 1, 1]
        categories: int tensor with shape [batch_size, seq_length]

    Returns: Weights of shape [batch_size * seq_length]

    """
    r = tf.range(0, limit=tf.shape(mask_positions)[0])
    r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])
    indices = tf.squeeze(tf.concat([r, mask_positions], axis=-1), axis=[1])
    updates = tf.ones(shape=(tf.shape(mask_positions)[0]))
    weights = tf.scatter_nd(indices, updates, tf.shape(categories))
    weights = tf.cast(weights, dtype="float32")
    weights = tf.reshape(weights, [-1])
    return weights


def xentropy_loss(y_pred, y_true, categories, mask_positions,
                  acc: tf.keras.metrics.CategoricalAccuracy = None,
                  debug: bool = False,
                  categorywise_only: bool = False):
    """
    Computes cross-entropy loss from the predictions
    
    Args:
        y_pred: Outputs of the encoder, float tensor of shape [batch_size, seq_length, hidden_size]
        y_true: Batch embedded with the preprocessor, float tensor of shape [batch_size, seq_length, hidden_size]
        categories: int tensor of shape [batch_size, seq_length]
        mask_positions: int tensor of shape [batch_size, 1, 1]
        acc: CategoricalAccuracy
        debug: Enable debug mode
        categorywise_only: Compute loss only from the products of same categories

    Returns: Mean cross-entropy loss of the predictions

    """
    logger = tf.get_logger()

    # Compute loss only from mask token
    weights = _get_mask_positions_weights(mask_positions,categories)
    weights_sum = tf.reduce_sum(weights)

    # Reshape to [batch_size * seq_length, hidden_size]
    hidden_size = y_pred.shape[2]
    pred_batch = tf.reshape(y_pred, [-1, hidden_size])
    true_batch = tf.reshape(y_true, [-1, hidden_size])
    item_count = tf.shape(true_batch)[0]

    # Dot product of every prediction with all labels
    logits = tf.matmul(pred_batch, true_batch, transpose_b=True)

    # Compute logits only within categories
    if categorywise_only:
        flat_categories = tf.reshape(categories, [-1])
        cat_mask = tf.equal(flat_categories[:, tf.newaxis], flat_categories[tf.newaxis, :])
        cat_mask = tf.logical_not(cat_mask)
        cat_mask = tf.cast(cat_mask, dtype="float32")
        cat_bias = cat_mask * _NEG_INF_FP32  # -inf on cells when categories don't match
        logits = tf.add(logits, cat_bias)
        if debug:
            logger.debug("Category bias")
            logger.debug(cat_bias)

    if debug:
        logger.debug("Loss weights")
        logger.debug(weights)
        logger.debug("Item Count")
        logger.debug(item_count)
        logger.debug("Logits")
        logger.debug(logits)

    # One-hot labels (indentity matrix)
    labels = tf.eye(item_count, item_count)

    if acc is not None:
        acc(labels, logits, sample_weight=weights)

    # Compute cross entropy on the masked positions
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy = tf.tensordot(cross_entropy, weights, 1)

    if debug:
        logger.debug("Cross Entropy")
        logger.debug(cross_entropy)

    # Return the mean of the losses
    return tf.reduce_sum(cross_entropy) / weights_sum


def fitb_dotproduct_acc(y_pred, y_true, pred_positions, target_position, pred_categories,
                        acc: tf.metrics.CategoricalAccuracy,
                        debug: bool = False):
    """
    Update FITB categorical accuracy

    Args:
        y_pred: Outputs of the encoder, float tensor of shape [1, seq_length, hidden_size]
        y_true: Candidates embedded with the preprocessor, float tensor of shape [1, candidates_count, hidden_size]
        pred_positions: positions of the predictions, int tensor of shape [1, 1, 1]
        target_position: position of the target, int tensor of shape [1]
        pred_categories: categories of the predictions, int tensor of shape [1, seq_length]
        acc: instance of CategoricalAccuracy
        debug: Enable debug mode

    Returns: Logits, float tensor of shape [seq_length, candidates_count]

    """
    logger = tf.get_logger()

    weights = _get_mask_positions_weights(pred_positions, pred_categories)

    if debug:
        logger.debug("Accuracy weights")
        logger.debug(weights)

    # Reshape to [seq length, hidden_size]
    hidde_size = y_pred.shape[2]
    pred_batch = tf.reshape(y_pred, [-1, hidde_size])
    true_batch = tf.reshape(y_true, [-1, hidde_size])

    # Dot product of every prediction with all labels
    logits = tf.matmul(pred_batch, true_batch, transpose_b=True)

    # Check that there is at least one non-zero value
    reduced_logits = tf.math.count_nonzero(logits)
    tf.debugging.assert_greater(reduced_logits, tf.constant([0], dtype="int64"), "There must at least one none zero "
                                                                                 "value in logits, otherwise the "
                                                                                 "accuracy doesn't work")

    # One-hot label: 1 on the position (pred_position, target_position), 0 on other positions
    pred_position = tf.squeeze(pred_positions, axis=[0, 1])
    target_position = tf.cast(target_position, dtype="int32")
    sparse_indices = tf.concat([pred_position, target_position], axis=0)
    sparse_indices = tf.expand_dims(sparse_indices, axis=0)
    labels = tf.scatter_nd(sparse_indices, tf.constant([1]), tf.shape(logits))

    if debug:
        logger.debug("Logits")
        logger.debug(logits)
        logger.debug("Labels")
        logger.debug(labels)

    if acc is not None:
        acc(labels, logits, sample_weight=weights)
    return logits


def get_distances(a, b):
    """
    Compute distances between vectors a and b

    Args:
        a: tensor of shape [a_count, hidden_size]
        b: tensor of shape [b_count, hidden_size]

    Returns: Euclidean distances to from of vectors a to b. Tensor of shape [a_count, b_count]

    """
    # Create tiles of shape [a_count, b_count, hidden_size]
    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)
    a = tf.tile(a, [1, b.shape[1], 1])
    b = tf.tile(b, [a.shape[0], 1, 1])

    sub = a - b  # d(a,b) = |a-b|
    return tf.math.reduce_euclidean_norm(sub, axis=-1)


def get_distances_to_targets(y_pred, y_true, pred_positions, target_positions, debug=False):
    """
    Compute distances to the correct targets and distances to the closest incorrect candidates

    y_pred and y_true have the same shapes during training
    Args:
        y_pred: float tensor of shape [batch_size, pred_seq_length, hidden_size]
        y_true: float tensor of shape [batch_size, true_seq_length, hidden_size]
        pred_positions: positions of the predictions, int tensor of shape [batch_size, 1, 1]
        target_positions: positions of the correct items, int tensor of shape [batch_size, 1, 1]
        debug: enable debug mode

    Returns: Pair (distances to positives, distances to negatives), both float tensors of shape [batch_size,]

    """
    logger = tf.get_logger()

    r = tf.range(0, limit=tf.shape(pred_positions)[0])
    r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])
    # indices of predictions [batch_size, 2]
    pred_indices = tf.squeeze(tf.concat([r, pred_positions], axis=-1), axis=[1])
    predictions = tf.gather_nd(y_pred, pred_indices)

    targets = tf.reshape(y_true, [-1, y_true.shape[-1]])
    dist = get_distances(predictions, targets)
    dist = tf.reshape(dist, (dist.shape[0], y_true.shape[0], -1))  # [batch_size, batch_size, seq_length]

    # indices of correct targets [batch_size, 2]
    target_indices = tf.squeeze(tf.concat([r, target_positions], axis=-1), axis=[1])
    dist_to_pos = tf.gather_nd(dist, target_indices, 1)  # [batch_size,]

    # Replace distances with true targets with max float, so they don't affect min aggregation
    r = tf.squeeze(r, axis=[1])
    target_indices = tf.concat([r, target_indices], axis=-1)  # [batch_size, 3]
    updates = tf.repeat(tf.constant(tf.float32.max, shape=1), target_indices.shape[0])
    dist_to_neg = tf.tensor_scatter_nd_update(dist, target_indices, updates)
    dist_to_neg = tf.math.reduce_min(dist_to_neg, axis=[-2, -1])  # [batch_size,]

    if debug:
        logger.debug("Predictions")
        logger.debug(predictions)
        logger.debug("Targets")
        logger.debug(targets)
        logger.debug("Distances")
        logger.debug(dist)
        logger.debug("Distances to positives")
        logger.debug(dist_to_pos)
        logger.debug("Distances to negatives")
        logger.debug(dist_to_neg)

    return dist_to_pos, dist_to_neg


def _update_distance_accuracy(acc: tf.keras.metrics.Accuracy, dist_to_pos, dist_to_neg, debug=False):
    """
    Update accuracy based on the distances to correct items and distances to the closest negative item

    Args:
        acc: instance of tf.keras.metrics.Accuracy
        dist_to_pos: flaot tensor of shape [batch_size,]
        dist_to_neg: flaot tensor of shape [batch_size,]
        debug: enable debug mode
    """
    # The predictions are correct, when minimal distance is to positives
    dist_delta = dist_to_pos - dist_to_neg
    correct = tf.math.less_equal(dist_delta, tf.zeros_like(dist_delta))  # correct when delta is less or equal to zero
    correct = tf.cast(correct, dtype=tf.int32)
    correct = tf.expand_dims(correct, axis=-1)
    acc.update_state(tf.ones_like(correct), correct)  # update the accuracy
    if debug:
        logger = tf.get_logger()
        logger.debug("Distance delta")
        logger.debug(dist_delta)
        logger.debug("Correct predictions")
        logger.debug(correct)


def distance_loss(y_pred, y_true, categories, mask_positions, margin,
                  acc: tf.metrics.Accuracy = None, debug=False):
    """
    Compute distance loss max(d(P, T) - n + m,0), where
        P is the prediction
        T is the true target
        n is the distance to the closest incorrect candidate
        m is a margin parameter

    Args:
        y_pred: float tensor of shape [batch_size, seq_length, hidden_size]
        y_true: float tensor of shape [batch_size, seq_length, hidden_size]
        mask_positions: positions of the mask tokens, int tensor of shape [batch_size, 1, 1]
        categories: int tensor of shape [batch_size, seq_length]
        margin: float parameter of the loss function
        acc: instance of tf.metrics.Accuracy
        debug: enable debug mode

    Returns: Float value of the distance loss function

    """
    logger = tf.get_logger()

    # Replace padding with max vectors
    padding_indices = tf.where(tf.math.equal(categories, tf.zeros_like(categories)))
    max_tensor = tf.repeat(tf.constant(_INF_FP32), [y_true.shape[-1]])
    max_tensor = tf.expand_dims(max_tensor, 0)
    max_tensor = tf.tile(max_tensor, [padding_indices.shape[0], 1])
    y_true = tf.tensor_scatter_nd_update(y_true, padding_indices, max_tensor)

    # Get distances to correct items and aggregated distances to negative items, both of shapes [batch_size,]
    dist_to_pos, dist_to_neg = get_distances_to_targets(y_pred, y_true, mask_positions, mask_positions, debug)

    if acc is not None:
        _update_distance_accuracy(acc, dist_to_pos, dist_to_neg, debug)

    # Repeat the margin for every outfit of the batch
    margin = tf.constant(margin)
    margin = tf.repeat(margin, dist_to_pos.shape[0])  # [batch_size,]

    # Compute the loss function
    loss = dist_to_pos - dist_to_neg + margin
    loss = tf.maximum(loss, 0)

    if debug:
        logger.debug("Distance losses")
        logger.debug(loss)

    return tf.reduce_mean(loss)


def fitb_distance_acc(y_pred, y_true, pred_positions, target_position, acc: tf.metrics.Accuracy, debug=False):
    """
    Updates the FITB accuracy using Euclidean distance to compute similarity

    Args:
        y_pred: float tensor of shape [1, pred_seq_length, hidden_size]
        y_true: float tensor of shape [1, true_seq_length, hidden_size]
        pred_positions: positions of the predictions, int tensor of shape [1, 1, 1]
        target_position: positions of the correct items, int tensor of shape [1]
        acc: instance of tf.metrics.Accuracy
        debug: enable debug mode
    """
    target_position = tf.cast(target_position, dtype="int32")
    target_position = tf.reshape(target_position, shape=[1, 1, 1])
    dist_to_pos, dist_to_neg = get_distances_to_targets(y_pred, y_true, pred_positions, target_position, debug)
    _update_distance_accuracy(acc, dist_to_pos, dist_to_neg)
