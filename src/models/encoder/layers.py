from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import src.models.encoder.utils as utils
import tensorflow as tf
from official.nlp import bert_modeling as common_layer


class SingleMasking(tf.keras.layers.Layer):

    def __init__(self, params):
        super(SingleMasking, self).__init__()
        self.params = params

        self.tokens_embedding = tf.keras.layers.Embedding(input_dim=1,
                                                          output_dim=self.params["hidden_size"],
                                                          name="tokens_embedding")
        self.token_id = tf.constant([0])
        self.dropout = tf.keras.layers.Dropout(self.params["emb_dropout"])

    def __call__(self, inputs, training=False, *args, **kwargs):
        inputs, categories, mask_positions = inputs[0], inputs[1], inputs[2]
        logger = tf.get_logger()

        with tf.name_scope("Masking"):
            mask_tensor = self.tokens_embedding(self.token_id)

            # Repeat the tensor_to_place to match the count of positions
            repeated_mask = tf.tile(mask_tensor, [tf.shape(mask_positions)[0], 1])
            if self.params["mode"] == "debug":
                logger.debug("Mask positions")
                logger.debug(mask_positions)
                logger.debug("Mask tensor")
                logger.debug(mask_tensor)
                logger.debug("repeated_mask")
                logger.debug(repeated_mask)
            # Reshape to (number of masked items, feature_dim)
            repeated_mask = tf.reshape(repeated_mask, shape=(-1, tf.shape(mask_tensor)[1]))

            if training:
                repeated_mask = self.dropout(repeated_mask)

            masked_inputs = utils.place_tensor_on_positions(inputs, repeated_mask, mask_positions, False)

        return masked_inputs


class CategoryMasking(tf.keras.layers.Layer):

    def __init__(self, params):
        super(CategoryMasking, self).__init__()
        self.params = params

        self.tokens_embedding = tf.keras.layers.Embedding(input_dim=params["categories_count"],
                                                          output_dim=self.params["hidden_size"],
                                                          name="tokens_embedding")

    def __call__(self, inputs, *args, **kwargs):
        inputs, categories, mask_positions = inputs[0], inputs[1], inputs[2]
        logger = tf.get_logger()

        with tf.name_scope("Masking"):
            cat_indices = tf.reshape(mask_positions, (-1, 1))
            mask_categories = tf.gather_nd(categories, mask_positions, batch_dims=1)
            mask_categories = tf.reshape(mask_categories, (-1, 1))
            if self.params["mode"] == "debug":
                logger.debug("Categories")
                logger.debug(categories)
                logger.debug("Category indices")
                logger.debug(cat_indices)
                logger.debug("Mask category")
                logger.debug(mask_categories)
            # Get mask token embeddings based on category
            mask_tensor = self.tokens_embedding(mask_categories)
            mask_tensor = tf.squeeze(mask_tensor, axis=[1])
            if self.params["mode"] == "debug":
                logger.debug("Mask tensor")
                logger.debug(mask_tensor)

            # Replace actual embedding with mask token embeddings
            masked_inputs = utils.place_tensor_on_positions(inputs, mask_tensor, mask_positions, repeated=False)

        return masked_inputs


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

        # if params["feature_dim"] != params["category_dim"]:
        #     raise RuntimeError("'feature_dim' must match 'category_dim' when category_merge is set to 'add' or "
        #                        "'multiply'")

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
            embedded_categories = tf.einsum("ij,jk->ik", mask_matrix, embedded_categories)  # Apply mask
            embedded_categories = tf.reshape(embedded_categories,  # Reshape to the initial shape
                                             shape=(batch_size, seq_length, self.params["category_dim"]))

            # Replace mask category embedding with zero tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                zero_tensor = tf.zeros(shape=(self.params["category_dim"],))
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

        # if params["feature_dim"] != params["category_dim"]:
        #     raise RuntimeError("'feature_dim' must match 'category_dim' when category_merge is set to 'add' or "
        #                        "'multiply'")

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
                                             shape=(batch_size, seq_length, self.params["category_dim"]))

            # Replace mask category embedding with ones tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                ones_tensor = tf.ones(shape=(self.params["category_dim"],))
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


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout

  def build(self, input_shape):
    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    size_per_head = self.hidden_size // self.num_heads
    self.query_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="query")
    self.key_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="key")
    self.value_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="value")
    self.output_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, output_projection=True, name="output_transform")
    super(Attention, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }

  def call(self, query_input, categories, source_input, bias, training, cache=None,
           decode_loop_step=None):
    """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]}
        where i is the current decoded length for non-padded decode, or max
        sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    logger = tf.get_logger()
    query = self.query_dense_layer(query_input)

    if categories is not None:
        key = self.key_dense_layer(categories)
        logger.debug("One hot categories")
        logger.debug(categories)
        logger.debug("Keys")
        logger.debug(key)
    else:
        key = self.key_dense_layer(source_input)

    value = self.value_dense_layer(source_input)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
        value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

    # Scale query to prevent the dot product between query and key from growing
    # too large.
    depth = (self.hidden_size // self.num_heads)
    query *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.einsum("BTNH,BFNH->BNFT", key, query)

    logits += bias
    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    weights = tf.nn.softmax(logits, name="attention_weights")

    if training:
      weights = tf.nn.dropout(weights, rate=self.attention_dropout)
    attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

    # Run the outputs through another linear projection layer. Recombining heads
    # is automatically done --> [batch_size, length, hidden_size]
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, query_input, categories, bias, training, cache=None,
           decode_loop_step=None):
    return super(SelfAttention, self).call(
        query_input, categories, query_input, bias, training, cache, decode_loop_step)