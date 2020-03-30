import tensorflow as tf

import src.models.transformer.utils as utils


class CategoryAdder(tf.keras.layers.Layer):

    def __init__(self, params):
        super(CategoryAdder, self).__init__()

        self.params = params

        if params["category_embedding"]:
            self.category_embedding = tf.keras.layers.Embedding(params["categories_count"],
                                                                output_dim=self.params["category_dim"],
                                                                name="category_embedding",
                                                                embeddings_initializer="zeros",
                                                                trainable=True)

        if params["feature_dim"] != params["category_dim"]:
            raise RuntimeError("'feature_dim' must match 'category_dim' when category_merge is set to 'add' or "
                               "'multiply'")

    def __call__(self, inputs, *args, **kwargs):
        with tf.name_scope("category-adder"):
            inputs, categories, mask_positions = inputs

            logger = tf.get_logger()

            flat_categories = tf.reshape(categories, shape=(-1, 1))

            mask_matrix = utils.compute_padding_mask_from_categories(categories)

            batch_size = tf.shape(inputs)[0]
            seq_length = tf.shape(inputs)[1]

            embedded_categories = self.category_embedding(flat_categories)
            embedded_categories = tf.squeeze(embedded_categories, axis=1)
            embedded_categories = tf.einsum("ij,jk->ik", mask_matrix, embedded_categories)
            embedded_categories = tf.reshape(embedded_categories,
                                             shape=(batch_size, seq_length, self.params["feature_dim"]))

            # Replace mask category embedding with zero tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                zero_tensor = tf.zeros(shape=(self.params["feature_dim"],))
                embedded_categories = utils.place_tensor_on_positions(embedded_categories, zero_tensor, mask_positions)

            if self.params["mode"] == "debug":
                logger.debug("Embedded categories")
                logger.debug(embedded_categories)

            return tf.keras.layers.add([inputs, embedded_categories])


class CategoryMultiplier(tf.keras.layers.Layer):

    def __init__(self, params):
        super(CategoryMultiplier, self).__init__()

        self.params = params

        if params["category_embedding"]:
            self.category_embedding = tf.keras.layers.Embedding(params["categories_count"],
                                                                output_dim=self.params["category_dim"],
                                                                name="category_embedding",
                                                                embeddings_initializer="ones",
                                                                trainable=True)

        if params["feature_dim"] != params["category_dim"]:
            raise RuntimeError("'feature_dim' must match 'category_dim' when category_merge is set to 'add' or "
                               "'multiply'")

    def __call__(self, inputs, *args, **kwargs):
        with tf.name_scope("category-multiplier"):
            inputs, categories, mask_positions = inputs

            logger = tf.get_logger()

            flat_categories = tf.reshape(categories, shape=(-1, 1))

            mask_matrix = utils.compute_padding_mask_from_categories(categories)

            batch_size = tf.shape(inputs)[0]
            seq_length = tf.shape(inputs)[1]

            embedded_categories = self.category_embedding(flat_categories)
            embedded_categories = tf.squeeze(embedded_categories, axis=1)
            embedded_categories = tf.einsum("ij,jk->ik", mask_matrix, embedded_categories)
            ones_tensor = tf.ones_like(embedded_categories)

            # Add ones to the place of zeroes - so it has no effect on visual embedding after multiplication
            inverted_mask = tf.reduce_sum(mask_matrix, axis=1)
            inverted_mask = tf.cast(inverted_mask, dtype="bool")
            inverted_mask = tf.logical_not(inverted_mask)
            inverted_mask = tf.cast(inverted_mask, dtype="float32")
            inverted_mask_matrix = tf.linalg.set_diag(mask_matrix, inverted_mask)
            ones_tensor = tf.einsum("ij,jk->ik", inverted_mask_matrix, ones_tensor)
            embedded_categories = tf.add(ones_tensor, embedded_categories)

            embedded_categories = tf.reshape(embedded_categories,
                                             shape=(batch_size, seq_length, self.params["feature_dim"]))

            # Replace mask category embedding with ones tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                ones_tensor = tf.ones(shape=(self.params["feature_dim"],))
                embedded_categories = utils.place_tensor_on_positions(embedded_categories, ones_tensor, mask_positions)

            if self.params["mode"] == "debug":
                logger.debug("Embedded categories")
                logger.debug(embedded_categories)

            return tf.keras.layers.multiply([inputs, embedded_categories])


class CategoryConcater(tf.keras.layers.Layer):

    def __init__(self, params):
        super(CategoryConcater, self).__init__()

        self.params = params

        if params["category_embedding"]:
            self.category_embedding = tf.keras.layers.Embedding(params["categories_count"],
                                                                output_dim=self.params["category_dim"],
                                                                name="category_embedding",
                                                                embeddings_initializer="truncated_normal",
                                                                trainable=True)
        if not self.params["with_mask_category_embedding"]:
            self.mask_category = tf.keras.layers.Embedding(input_dim=1,
                                                              output_dim=self.params["category_dim"],
                                                              name="mask_category")
        self.mask_category_id = tf.constant([0])

    def __call__(self, inputs, *args, **kwargs):
        with tf.name_scope("category-concater"):
            inputs, categories, mask_positions = inputs

            logger = tf.get_logger()

            flat_categories = tf.reshape(categories, shape=(-1, 1))

            mask_matrix = utils.compute_padding_mask_from_categories(categories)

            batch_size = tf.shape(inputs)[0]
            seq_length = tf.shape(inputs)[1]

            embedded_categories = self.category_embedding(flat_categories)
            embedded_categories = tf.squeeze(embedded_categories, axis=1)
            embedded_categories = tf.einsum("ij,jk->ik", mask_matrix, embedded_categories)
            embedded_categories = tf.reshape(embedded_categories,
                                             shape=(batch_size, seq_length, self.params["category_dim"]))

            # Replace mask category embedding with zero tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                mask_category = self.mask_category(self.mask_category_id)
                mask_category = tf.squeeze(mask_category)
                embedded_categories = utils.place_tensor_on_positions(embedded_categories, mask_category, mask_positions)

            if self.params["mode"] == "debug":
                logger.debug("Embedded categories")
                logger.debug(embedded_categories)

            return tf.keras.layers.concatenate([inputs, embedded_categories])