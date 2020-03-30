import tensorflow as tf


def build_category_lookup_table():
    categories = {
        "top": [11, 15, 17, 18, 19, 21, 343, 104, 236, 247, 252, 272, 273, 275, 286, 309, 342, 4454, 4495, 4496, 4497,
                4498
            , 341],
        "bottom": [7, 8, 9, 10, 27, 28, 29, 237, 238, 239, 240, 241, 253, 254, 255, 278, 279, 280, 287, 288,
                   310, 4458, 4459],
        "shoes": [41, 42, 43, 46, 47, 48, 49, 50, 261, 262, 263, 264, 265, 266, 267, 268, 291, 292, 293, 294, 295, 296,
                  297, 298, 4464, 4465, 4522],
        "accessories": [35, 36, 37, 38, 39, 40, 51, 52, 53, 55, 56, 57, 58, 59, 105, 231, 258, 259, 260, 270,
                        290, 299, 300, 301, 302, 303, 304, 306, 4428, 4426, 4447, 4461, 4462, 4463, 4468, 4470, 4472
            , 4473, 4474, 4520, 4521, ],
        "jewelry": [60, 61, 62, 64, 65, 67, 106, 107, 305, 307, 4466, 4467, 4523, 4524, 4525, ],
        "other-wearable": [2, 31, 33, 68, 69, 71, 85, 108, 245, 246, 248, 249, 251, 257, 271, 282, 283, 284, 285,
                           4460, 4517, 4518, 1605, 1606],
        "full": [3, 4, 5, 6, 30, 75, 243, 244, 250, 281, 4516],
        "outerwear": [23, 24, 25, 26, 256, 276, 277, 289, 4455, 4456, 4457, ]
    }

    keys = []
    values = []
    cat_group = 1
    for category in categories.values():
        for cat_id in category:
            keys.append(cat_id)
            values.append(cat_group)
        cat_group += 1

    keys_tensor = tf.constant(keys, dtype="int64")
    vals_tensor = tf.constant(values, dtype="int64")
    return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), len(categories)+1)


def compute_padding_mask_from_categories(categories):
    """

    Args:
        categories: Tensor of shape (batch_size, sequence_length),

    Returns:
        Matrix of shape (batch_size*sequence_length, batch_size*sequence_length) with zeroes except for diagonal.
            There's 0 on the diagonal if the corresponding item should be masked
            There's 1 on the diagonal if the corresponding item should not be masked
    """
    # Compute padding mask # TODO: Make function out of this
    unpacked_categories = tf.reshape(categories, shape=[-1])
    unpacked_length = tf.shape(unpacked_categories)[0]
    padding_mask = tf.equal(unpacked_categories, 0)
    # Category 0 is considered as masked - category embedding is not applied
    padding_mask = tf.math.logical_not(padding_mask)
    padding_mask = tf.cast(padding_mask, dtype="float32")
    mask_matrix = tf.zeros(shape=(unpacked_length, unpacked_length))
    return tf.linalg.set_diag(mask_matrix, padding_mask)


def place_tensor_on_positions(inputs, tensor_to_place, positions):
    # Repeat the tensor_to_place to match the count of positions
    repeated = tf.repeat(tensor_to_place, tf.shape(positions)[0])
    # Reshape to (number of masked items, feature_dim)
    repeated = tf.reshape(repeated, shape=(-1, tf.shape(tensor_to_place)[0]))
    r = tf.range(0, limit=tf.shape(positions)[0], dtype="int32")
    r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])
    indices = tf.concat([r, positions], axis=-1)
    indices = tf.squeeze(indices, axis=[1])
    return tf.tensor_scatter_nd_update(inputs, indices, repeated)
