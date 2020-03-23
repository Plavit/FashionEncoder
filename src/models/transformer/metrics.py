import tensorflow as tf


def xentropy_loss(y_pred, y_true, categories, mask_positions, acc=None):
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

    logger.debug("Loss weights")
    logger.debug(weights)

    # Reshape to batch (size * seq length, feature dim)
    pred_batch = tf.reshape(y_pred, [-1, feature_dim])
    true_batch = tf.reshape(y_true, [-1, feature_dim])
    item_count = tf.shape(true_batch)[0]
    logger.debug("Item Count")
    logger.debug(item_count)

    # Dot product of every prediction with all labels
    logits = tf.matmul(pred_batch, true_batch, transpose_b=True)

    logger.debug("Logits")
    logger.debug(logits)

    # One-hot labels (the indentity matrix)
    labels = tf.eye(item_count, item_count)

    if acc is not None:
        acc(labels, logits, sample_weight=weights)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    logger.debug(cross_entropy)
    cross_entropy = tf.tensordot(cross_entropy, weights, 1)
    logger.debug("Cross Entropy")
    logger.debug(cross_entropy)
    return tf.reduce_sum(cross_entropy) / weights_sum


def fitb_acc(y_pred, y_true, pred_positions, target_position, categories, acc: tf.metrics.CategoricalAccuracy):
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
