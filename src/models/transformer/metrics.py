import tensorflow as tf


def xentropy_loss(y_pred, y_true, categories, mask_positions, acc=None):
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

    logger = tf.get_logger()
    logger.debug(weights)

    # Reshape to batch (size * seq length, feature dim)
    pred_batch = tf.reshape(y_pred, [-1, feature_dim])
    true_batch = tf.reshape(y_true, [-1, feature_dim])
    item_count = true_batch.shape[0]

    # Dot product of every prediction with all labels
    logits = tf.matmul(pred_batch, true_batch, transpose_b=True)
    logger.debug(logits)

    # One-hot labels (the indentity matrix)
    labels = tf.eye(item_count, item_count)

    # Compute weights to identify padding values
    # not_padded = tf.equal(true_batch, 0)
    # not_padded = tf.reduce_all(not_padded, axis=1)
    # not_padded = tf.math.logical_not(not_padded)
    # not_padded = tf.cast(not_padded, dtype="float32")
    # items_in_batch = tf.reduce_sum(not_padded)

    if acc is not None:
        acc(labels, logits, sample_weight=weights)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    logger.debug(cross_entropy)
    cross_entropy = tf.tensordot(cross_entropy, weights, 1)
    logger.debug(cross_entropy)
    return tf.reduce_sum(cross_entropy) / weights_sum


def categorical_acc(y_pred, y_true, acc: tf.metrics.CategoricalAccuracy):
    feature_dim = y_pred.shape[2]

    # Reshape to batch (size * seq length, feature dim)
    pred_batch = tf.reshape(y_pred, [-1, feature_dim])
    true_batch = tf.reshape(y_true, [-1, feature_dim])
    item_count = true_batch.shape[0]

    # Dot product of every prediction with all labels
    logits = tf.matmul(pred_batch, true_batch, transpose_b=True)

    # One-hot labels (the indentity matrix)

    labels = tf.eye(item_count, item_count)

    # Compute weights to identify padding values
    weights = tf.equal(true_batch, 0)
    weights = tf.reduce_all(weights, axis=1)
    weights = tf.math.logical_not(weights)
    weights = tf.cast(weights, dtype="float32")

    acc(labels, logits, sample_weight=weights)
