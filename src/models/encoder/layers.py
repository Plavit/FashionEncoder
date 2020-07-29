from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import src.models.encoder.utils as utils


class SingleMasking(tf.keras.layers.Layer):
    """A layer providing single-token masking"""

    def __init__(self, params):
        super(SingleMasking, self).__init__()
        self.params = params

        # With category concatenation the size of mask is smaller than hidden size
        if "category_embedding" in params and params["category_merge"] == "concat":
            size = params["hidden_size"] - params["category_dim"]
        else:
            size = params["hidden_size"]

        self.tokens_embedding = tf.keras.layers.Embedding(input_dim=1,
                                                          output_dim=size,
                                                          name="tokens_embedding")
        self.token_id = tf.constant([0])
        self.dropout = tf.keras.layers.Dropout(self.params["emb_dropout"])

    def __call__(self, inputs, training=False, *args, **kwargs):
        """
        Replace the vectors on mask_positions with the mask token

        Args:
            inputs: input tensor list of size 3
                First item, inputs: float tensor with shape [batch_size, seq_length, feature_dim]
                Second item, categories: int tensor with shape [batch_size, seq_length].
                Third item, mask positions: int tensor with shape [batch_size, 1, 1]
            training: boolean, whether in training mode or not.

        Returns: Masked inputs float tensor with shape [batch_size, seq_length, feature_dim]

        """
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

            # Replace the vectors on mask positions with the mask token prepared in repeated_mask
            masked_inputs = utils.place_tensor_on_positions(inputs, repeated_mask, mask_positions, False)

        return masked_inputs


class CategoryMasking(tf.keras.layers.Layer):
    """A layer providing category-specific masking"""

    def __init__(self, params):
        super(CategoryMasking, self).__init__()
        self.params = params

        # With category concatenation the size of mask is smaller than hidden size
        if "category_embedding" in params and params["category_merge"] == "concat":
            size = params["hidden_size"] - params["category_dim"]
        else:
            size = params["hidden_size"]

        self.tokens_embedding = tf.keras.layers.Embedding(input_dim=params["categories_count"],
                                                          output_dim=size,
                                                          name="tokens_embedding")

    def __call__(self, inputs, *args, **kwargs):
        """
        Replace the vectors on mask_positions with the mask token that is learned for each category

        Args:
            inputs: input tensor list of size 3
                First item, inputs: float tensor with shape [batch_size, seq_length, feature_dim]
                Second item, categories: int tensor with shape [batch_size, seq_length].
                Third item, mask positions: int tensor with shape [batch_size, 1, 1]
            training: boolean, whether in training mode or not.

        Returns: Masked inputs float tensor with shape [batch_size, seq_length, feature_dim]

        """
        inputs, categories, mask_positions = inputs[0], inputs[1], inputs[2]
        logger = tf.get_logger()

        with tf.name_scope("Masking"):
            # Get categories on masked positions
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

            # Get mask token embeddings based on the categories
            mask_tensor = self.tokens_embedding(mask_categories)
            mask_tensor = tf.squeeze(mask_tensor, axis=[1])

            if self.params["mode"] == "debug":
                logger.debug("Mask tensor")
                logger.debug(mask_tensor)

            # Replace actual embedding with mask token embeddings
            masked_inputs = utils.place_tensor_on_positions(inputs, mask_tensor, mask_positions, repeat=False)

        return masked_inputs


class CategoryAdder(tf.keras.layers.Layer):
    """A layer that adds category embedding to the inputs"""

    def __init__(self, params):
        super(CategoryAdder, self).__init__()

        self.params = params

        if params["category_embedding"]:
            self.category_embedding = tf.keras.layers.Embedding(params["categories_count"],
                                                                output_dim=self.params["category_dim"],
                                                                name="category_embedding",
                                                                embeddings_initializer="zeros",
                                                                trainable=True)

    def __call__(self, inputs, *args, **kwargs):
        """
        Replace the vectors on mask_positions with the mask token that is learned for each category

        Args:
            inputs: input tensor list of size 3
                First item, inputs: float tensor with shape [batch_size, seq_length, feature_dim]
                Second item, categories: int tensor with shape [batch_size, seq_length].
                Third item, mask positions: int tensor with shape [batch_size, 1, 1]

        Returns: Inputs with added category - float tensor with shape [batch_size, seq_length, feature_dim]

        """

        with tf.name_scope("category-adder"):
            inputs, categories, mask_positions = inputs

            logger = tf.get_logger()

            embedded_categories = utils.get_category_embedding(categories, self.category_embedding, "zeros")

            # Optionally replace mask category embedding with zero tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                zero_tensor = tf.zeros(shape=(self.params["category_dim"],))
                embedded_categories = utils.place_tensor_on_positions(embedded_categories, zero_tensor, mask_positions)

            if self.params["mode"] == "debug":
                logger.debug("Embedded categories")
                logger.debug(embedded_categories)

            return tf.keras.layers.add([inputs, embedded_categories])


class CategoryMultiplier(tf.keras.layers.Layer):
    """A layer that multiplies the category embedding with the inputs"""

    def __init__(self, params):
        super(CategoryMultiplier, self).__init__()

        self.params = params

        if params["category_embedding"]:
            self.category_embedding = tf.keras.layers.Embedding(params["categories_count"],
                                                                output_dim=self.params["category_dim"],
                                                                name="category_embedding",
                                                                embeddings_initializer="ones",
                                                                trainable=True)

    def __call__(self, inputs, *args, **kwargs):
        """
        Replace the vectors on mask_positions with the mask token that is learned for each category

        Args:
            inputs: input tensor list of size 3
                First item, inputs: float tensor with shape [batch_size, seq_length, feature_dim]
                Second item, categories: int tensor with shape [batch_size, seq_length].
                Third item, mask positions: int tensor with shape [batch_size, 1, 1]

        Returns: Inputs multiplied by category embedding - float tensor with shape [batch_size, seq_length, feature_dim]

        """
        with tf.name_scope("category-multiplier"):
            inputs, categories, mask_positions = inputs

            logger = tf.get_logger()

            embedded_categories = utils.get_category_embedding(categories, self.category_embedding, "ones")

            # Replace mask category embedding with ones tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                ones_tensor = tf.ones(shape=(self.params["category_dim"],))
                embedded_categories = utils.place_tensor_on_positions(embedded_categories, ones_tensor, mask_positions)

            if self.params["mode"] == "debug":
                logger.debug("Embedded categories")
                logger.debug(embedded_categories)

            return tf.keras.layers.multiply([inputs, embedded_categories])


class CategoryConcater(tf.keras.layers.Layer):
    """A layer that concatenates the category embedding to the inputs"""

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
        """
        Replace the vectors on mask_positions with the mask token that is learned for each category

        Args:
            inputs: input tensor list of size 3
                First item, inputs: float tensor with shape [batch_size, seq_length, feature_dim]
                Second item, categories: int tensor with shape [batch_size, seq_length].
                Third item, mask positions: int tensor with shape [batch_size, 1, 1]

        Returns: Inputs with concatenated category embedding
            float tensor with shape [batch_size, seq_length, feature_dim]

        """
        with tf.name_scope("category-concater"):
            inputs, categories, mask_positions = inputs

            logger = tf.get_logger()

            embedded_categories = utils.get_category_embedding(categories, self.category_embedding, "zeros")

            # Optionally replace mask category embedding with zero tensor
            if not self.params["with_mask_category_embedding"] and mask_positions is not None:
                mask_category = self.mask_category(self.mask_category_id)
                mask_category = tf.squeeze(mask_category)
                embedded_categories = utils.place_tensor_on_positions(embedded_categories, mask_category,
                                                                      mask_positions)

            if self.params["mode"] == "debug":
                logger.debug("Embedded categories")
                logger.debug(embedded_categories)

            return tf.keras.layers.concatenate([inputs, embedded_categories])


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer.
    Implementation from Tensorflow Official Models (slightly modified)
    """

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
        self.query_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, name="query")
        self.key_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, name="key")
        self.value_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, name="value")
        self.output_dense_layer = Dense3D(
            self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
            use_bias=False, output_projection=True, name="output_transform")
        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
        }

    def call(self, query_input, categories, source_input, bias, training, cache=None, decode_loop_step=None):
        """Apply attention mechanism to query_input and source_input.

        Args:
          query_input: A tensor with shape [batch_size, length_query, hidden_size].
          categories: A tensor with shape [batch_size, seq_length]
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

        if categories is not None:
            key = self.key_dense_layer(categories)
            query = self.query_dense_layer(categories)
        else:
            key = self.key_dense_layer(source_input)
            query = self.query_dense_layer(query_input)

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
    """Multiheaded self-attention layer.
    Implementation based on Tensorflow Official Models
    """

    def call(self, query_input, categories, bias, training, cache=None,
             decode_loop_step=None):
        return super(SelfAttention, self).call(
            query_input, categories, query_input, bias, training, cache, decode_loop_step)


class Dense3D(tf.keras.layers.Layer):
    """A Dense Layer using 3D kernel with tf.einsum implementation.
    Implementation from Tensorflow Official Models - NOT MODIFIED

    Attributes:
      num_attention_heads: An integer, number of attention heads for each
        multihead attention layer.
      size_per_head: An integer, hidden size per attention head.
      hidden_size: An integer, dimension of the hidden layer.
      kernel_initializer: An initializer for the kernel weight.
      bias_initializer: An initializer for the bias.
      activation: An activation function to use. If nothing is specified, no
        activation is applied.
      use_bias: A bool, whether the layer uses a bias.
      output_projection: A bool, whether the Dense3D layer is used for output
        linear projection.
      backward_compatible: A bool, whether the variables shape are compatible
        with checkpoints converted from TF 1.x.
    """

    def __init__(self,
                 num_attention_heads=12,
                 size_per_head=72,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                 activation=None,
                 use_bias=True,
                 output_projection=False,
                 backward_compatible=False,
                 **kwargs):
        """Inits Dense3D."""
        super(Dense3D, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.hidden_size = num_attention_heads * size_per_head
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.output_projection = output_projection
        self.backward_compatible = backward_compatible

    @property
    def compatible_kernel_shape(self):
        if self.output_projection:
            return [self.hidden_size, self.hidden_size]
        return [self.last_dim, self.hidden_size]

    @property
    def compatible_bias_shape(self):
        return [self.hidden_size]

    @property
    def kernel_shape(self):
        if self.output_projection:
            return [self.num_attention_heads, self.size_per_head, self.hidden_size]
        return [self.last_dim, self.num_attention_heads, self.size_per_head]

    @property
    def bias_shape(self):
        if self.output_projection:
            return [self.hidden_size]
        return [self.num_attention_heads, self.size_per_head]

    def build(self, input_shape):
        """Implements build() for the layer."""
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `Dense3D` layer with non-floating "
                            "point (and non-complex) dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to `Dense3D` "
                             "should be defined. Found `None`.")
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: self.last_dim})
        # Determines variable shapes.
        if self.backward_compatible:
            kernel_shape = self.compatible_kernel_shape
            bias_shape = self.compatible_bias_shape
        else:
            kernel_shape = self.kernel_shape
            bias_shape = self.bias_shape

        self.kernel = self.add_weight(
            "kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        super(Dense3D, self).build(input_shape)

    def call(self, inputs):
        """Implements ``call()`` for Dense3D.

        Args:
          inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
            when output_projection is False, otherwise a float tensor of shape
            [batch_size, sequence_length, num_heads, dim_per_head].

        Returns:
          The projected tensor with shape [batch_size, sequence_length, num_heads,
            dim_per_head] when output_projection is False, otherwise [batch_size,
            sequence_length, hidden_size].
        """
        if self.backward_compatible:
            kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
            bias = (tf.keras.backend.reshape(self.bias, self.bias_shape)
                    if self.use_bias else None)
        else:
            kernel = self.kernel
            bias = self.bias

        if self.output_projection:
            ret = tf.einsum("abcd,cde->abe", inputs, kernel)
        else:
            ret = tf.einsum("abc,cde->abde", inputs, kernel)
        if self.use_bias:
            ret += bias
        if self.activation is not None:
            return self.activation(ret)
        return ret


class FeedForwardNetwork(tf.keras.layers.Layer):
    """Fully connected feedforward network.
    Implementation from Tensorflow Official Models - NOT MODIFIED
    """

    def __init__(self, hidden_size, filter_size, relu_dropout):
        """Initialize FeedForwardNetwork.

        Args:
          hidden_size: int, output dim of hidden layer.
          filter_size: int, filter size for the inner (first) dense layer.
          relu_dropout: float, dropout rate for training.
        """
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

    def build(self, input_shape):
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.filter_size,
            use_bias=True,
            activation=tf.nn.relu,
            name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=True, name="output_layer")
        super(FeedForwardNetwork, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          training: boolean, whether in training mode or not.

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """

        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)

        return output
