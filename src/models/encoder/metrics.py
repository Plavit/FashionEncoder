import tensorflow as tf


_NEG_INF_FP32 = -1e9
_INF_FP32 = 1e9


def xentropy_loss(y_pred, y_true, categories, mask_positions, acc=None, debug=False, categorywise_only=False):
    logger = tf.get_logger()

    feature_dim = y_pred.shape[2]

    # Compute loss only from mask token
    r = tf.range(0, limit=tf.shape(mask_positions)[0])
    r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])
    indices = tf.squeeze(tf.concat([r, mask_positions], axis=-1), axis=[1])
    updates = tf.ones(shape=(tf.shape(mask_positions)[0]))
    weights = tf.scatter_nd(indices, updates, tf.shape(categories))
    weights = tf.cast(weights, dtype="float32")
    weights = tf.reshape(weights, [-1])
    weights_sum = tf.reduce_sum(weights)

    # Reshape to batch (size * seq length, feature dim)
    pred_batch = tf.reshape(y_pred, [-1, feature_dim])
    true_batch = tf.reshape(y_true, [-1, feature_dim])
    item_count = tf.shape(true_batch)[0]

    # Dot product of every prediction with all labels
    logits = tf.matmul(pred_batch, true_batch, transpose_b=True)

    # Compute logits only within categories
    if categorywise_only:
        flat_categories = tf.reshape(categories, [-1])
        cat_mask = tf.equal(flat_categories[:, tf.newaxis], flat_categories[tf.newaxis, :])
        cat_mask = tf.logical_not(cat_mask)
        cat_mask = tf.cast(cat_mask, dtype="float32")
        cat_mask = cat_mask * _NEG_INF_FP32  # -inf on cells when categories don't match
        logits = tf.add(logits, cat_mask)
        if debug:
            logger.debug("Category mask")
            logger.debug(cat_mask)

    if debug:
        logger.debug("Loss weights")
        logger.debug(weights)
        logger.debug("Item Count")
        logger.debug(item_count)
        logger.debug("Logits")
        logger.debug(logits)

    # One-hot labels (the indentity matrix)
    labels = tf.eye(item_count, item_count)

    if acc is not None:
        acc(labels, logits, sample_weight=weights)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if debug:
        logger.debug(cross_entropy)
    cross_entropy = tf.tensordot(cross_entropy, weights, 1)
    if debug:
        logger.debug("Cross Entropy")
        logger.debug(cross_entropy)
    return tf.reduce_sum(cross_entropy) / weights_sum


def fitb_acc(y_pred, y_true, pred_positions, target_position, categories, acc: tf.metrics.CategoricalAccuracy,
             debug=False):
    logger = tf.get_logger()
    feature_dim = y_pred.shape[2]

    # Compute loss only from mask token
    r = tf.range(0, limit=tf.shape(pred_positions)[0])
    r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])
    indices = tf.squeeze(tf.concat([r, pred_positions], axis=-1), axis=[1])
    updates = tf.ones(shape=(tf.shape(pred_positions)[0]))
    weights = tf.scatter_nd(indices, updates, tf.shape(categories))
    weights = tf.cast(weights, dtype="float32")
    weights = tf.reshape(weights, [-1])

    logger.debug("Loss weights")
    logger.debug(weights)

    # Reshape to batch (size * seq length, feature dim)
    pred_batch = tf.reshape(y_pred, [-1, feature_dim])
    true_batch = tf.reshape(y_true, [-1, feature_dim])

    # Dot product of every prediction with all labels
    logits = tf.matmul(pred_batch, true_batch, transpose_b=True)

    reduced_logits = tf.math.count_nonzero(logits)
    tf.debugging.assert_greater(reduced_logits, tf.constant([0], dtype="int64"), "There must at least one none zero "
                                                                                 "value in logits, otherwise the "
                                                                                 "accuracy doesn't work")

    logger.debug("Logits")
    logger.debug(logits)

    pred_position = tf.squeeze(pred_positions, axis=[0, 1])
    # One-hot labels (the indentity matrix)
    target_position = tf.cast(target_position, dtype="int32")
    sparse_indices = tf.concat([pred_position, target_position], axis=0)
    sparse_indices = tf.expand_dims(sparse_indices, axis=0)

    labels = tf.scatter_nd(sparse_indices, tf.constant([1]), tf.shape(logits))
    logger.debug("Labels")
    logger.debug(labels)
    if acc is not None:
        acc(labels, logits, sample_weight=weights)
    return logits


def distances(a, b):
    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)

    a = tf.tile(a, [1, b.shape[1], 1])
    b = tf.tile(b, [a.shape[0], 1, 1])

    sub = a - b
    return tf.math.reduce_euclidean_norm(sub, -1)


def get_distances_to_targets(y_pred, y_true, mask_positions, debug=False):
    logger = tf.get_logger()

    r = tf.range(0, limit=tf.shape(mask_positions)[0])
    r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])

    # indices of mask tokens (batch_size, 2)
    indices = tf.squeeze(tf.concat([r, mask_positions], axis=-1), axis=[1])

    predictions = tf.gather_nd(y_pred, indices)

    targets = tf.reshape(y_true, [-1, y_true.shape[-1]])
    dist = distances(predictions, targets)
    dist = tf.reshape(dist, (dist.shape[0], y_true.shape[0], -1))  # (batch_size, batch_size, seq_length)

    dist_to_pos = tf.gather_nd(dist, indices, 1)  # (batch_size,)

    # Replace distances with true targets with max float, so it don't affect min aggregation
    r = tf.squeeze(r, axis=[1])
    indices = tf.concat([r, indices], axis=-1)  # (batch_size, 3)
    updates = tf.repeat(tf.constant(tf.float32.max, shape=1), indices.shape[0])
    dist_to_neg = tf.tensor_scatter_nd_update(dist, indices, updates)
    dist_to_neg = tf.math.reduce_min(dist_to_neg, axis=[-2, -1])  # (batch_size,)
    # TODO: Mean aggregation?

    if debug:
        logger.debug("Predictions")
        logger.debug(predictions)
        logger.debug("Distances")
        logger.debug(dist)
        logger.debug("Distances to positives")
        logger.debug(dist_to_pos)
        logger.debug("Distances to negatives")
        logger.debug(dist_to_neg)

    return dist_to_pos, dist_to_neg


def outfit_distance_loss(y_pred, y_true, categories, mask_positions, margin, acc: tf.metrics.Accuracy = None,
                         debug=False):
    logger = tf.get_logger()

    # Replace padding to max vectors
    padding_indices = tf.where(tf.math.equal(categories, tf.zeros_like(categories)))
    max_tensor = tf.repeat(tf.constant(_INF_FP32), [y_true.shape[-1]])
    max_tensor = tf.expand_dims(max_tensor, 0)
    max_tensor = tf.tile(max_tensor, [padding_indices.shape[0], 1])
    y_true = tf.tensor_scatter_nd_update(y_true, padding_indices, max_tensor)

    dist_to_pos, dist_to_neg = get_distances_to_targets(y_pred, y_true, mask_positions, debug)

    if acc is not None:
        # The predictions are correct, when minimal distance is to positives
        dist_delta = dist_to_pos - dist_to_neg
        correct = tf.math.less_equal(dist_delta, tf.zeros_like(dist_delta))
        correct = tf.cast(correct, dtype=tf.int32)
        correct = tf.expand_dims(correct, axis=-1)
        if debug:
            logger.debug("Distance delta")
            logger.debug(dist_delta)
            logger.debug("Correct predictions")
            logger.debug(correct)

        acc.update_state(tf.ones_like(correct), correct)

    margin = tf.constant(margin)
    margin = tf.repeat(margin, dist_to_pos.shape[0])

    loss = dist_to_pos - dist_to_neg + margin
    loss = tf.clip_by_value(loss, 0, tf.float32.max)

    if debug:
        logger.debug("Distance losses")
        logger.debug(loss)

    return tf.reduce_mean(loss)


def outfit_distance_fitb(y_pred, y_true, pred_positions, target_position, categories, acc: tf.metrics.Accuracy,
                         debug=False):
    dist_to_pos, dist_to_neg = get_distances_to_targets(y_pred, y_true, pred_positions, debug)

    if acc is not None:
        # The predictions are correct, when minimal distance is to positives
        dist_delta = dist_to_pos - dist_to_neg
        correct = tf.math.less_equal(dist_delta, tf.zeros_like(dist_delta))
        correct = tf.cast(correct, dtype=tf.int32)
        correct = tf.expand_dims(correct, axis=-1)
        acc.update_state(tf.ones_like(correct), correct)
