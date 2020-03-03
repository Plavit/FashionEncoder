import tensorflow as tf


def xentropy_loss(y_pred, y_true):
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
    items_in_batch = tf.reduce_sum(weights)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy = tf.tensordot(cross_entropy, weights, 1)
    return tf.reduce_sum(cross_entropy) / items_in_batch


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
    items_in_batch = tf.reduce_sum(weights)

    acc.update_state(labels, logits, sample_weight=weights)
