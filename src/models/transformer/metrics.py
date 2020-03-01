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
    print(item_count)
    labels = tf.eye(item_count, item_count)

    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels, logits))