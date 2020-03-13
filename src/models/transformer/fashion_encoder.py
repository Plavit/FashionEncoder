# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines the Fashion Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.transformer.model import model_utils
from official.transformer.utils.tokenizer import EOS_ID
from official.transformer.v2 import attention_layer
from official.transformer.v2 import beam_search
from official.transformer.v2 import ffn_layer

# Disable the not-callable lint error, since it claims many objects are not
# callable when they actually are.
# pylint: disable=not-callable


def create_model(params, is_train):
  """Creates transformer model."""
  with tf.name_scope("model"):
    categories = tf.keras.layers.Input((None,), dtype="int32", name="categories")
    mask_positions = tf.keras.layers.Input((None, 1), dtype="int32", name="mask_positions")

    if params["with_cnn"]:
        inputs = tf.keras.layers.Input((None, 299, 299, 3), dtype="float32", name="inputs")
        cnn = CNNExtractor(params, "inception_extractor")
        encoder_inputs = cnn([inputs, categories, mask_positions])
        cnn.summary()
    else:
        inputs = tf.keras.layers.Input((None, params["feature_dim"]), dtype="float32", name="inputs")
        encoder_inputs = inputs

    internal_model = FashionEncoder(params, name="fashion_encoder")

    if is_train:
        ret = internal_model([encoder_inputs, categories, mask_positions], training=True)
    else:
        ret = internal_model([encoder_inputs, categories, mask_positions], training=False)

    internal_model.summary()
    return tf.keras.Model([inputs, categories, mask_positions], [ret, encoder_inputs])  # TODO: Change for predict


class CNNExtractor(tf.keras.Model):

  def __init__(self, params, name=None):
    """Initialize layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      name: name of the model.
    """
    super(CNNExtractor, self).__init__(name=name)
    self.params = params
    # self.fashion_encoder = FashionEncoder(params, name="fashion_encoder")
    self.cnn_model = tf.keras.applications.inception_v3.InceptionV3(  # type: tf.keras.models.Model
            weights='imagenet',
            include_top=False,
            pooling='avg')
    for layer in self.cnn_model.layers[:249]:
        layer.trainable = False
    for layer in self.cnn_model.layers[249:]:
        layer.trainable = True

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, inputs):
      """

      Args:
          inputs: tuple (inputs, categories, mask_positions)
            inputs - tensor of shape (batch_size, seq_length, 299, 299, 3)
          training:
      """
      logger = tf.get_logger()

      inputs, categories, mask_positions = inputs[0], inputs[1], inputs[2]
      logger.debug(inputs)
      # Compute padding mask
      unpacked_categories = tf.reshape(categories, shape=[-1])
      unpacked_length = tf.shape(unpacked_categories)[0]
      padding_mask = tf.equal(unpacked_categories, 0)
      padding_mask = tf.math.logical_not(padding_mask)
      padding_mask = tf.cast(padding_mask, dtype="float32")
      mask_matrix = tf.zeros(shape=(unpacked_length, unpacked_length))
      mask_matrix = tf.linalg.set_diag(mask_matrix, padding_mask)

      batch_size = tf.shape(inputs)[0]
      seq_length = tf.shape(inputs)[1]

      inputs = tf.reshape(inputs, shape=(-1, 299, 299, 3))
      cnn_outputs = self.cnn_model(inputs)
      cnn_outputs = tf.einsum("ij,jk->ik", mask_matrix, cnn_outputs)
      logger.debug(cnn_outputs)
      logger.debug(cnn_outputs)
      return tf.reshape(cnn_outputs, shape=(batch_size, seq_length, self.params["feature_dim"]))


class FashionEncoder(tf.keras.Model):
  """Transformer model with Keras.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is a sequence of image features and sequence of
  categories (or a batch of these sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.

  """

  def __init__(self, params, name=None):
    """Initialize layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      name: name of the model.
    """
    super(FashionEncoder, self).__init__(name=name)
    self.params = params

    self.tokens_embedding = tf.keras.layers.Embedding(input_dim=1,
                                                      output_dim=self.params["feature_dim"],
                                                      name="tokens_embedding")

    i_dense = tf.keras.layers.Dense(self.params["hidden_size"], activation="relu",
                                    input_shape=(None, None, self.params["feature_dim"]), name="dense_input")
    self.input_dense = DenseLayerWrapper(i_dense, params)
    self.encoder_stack = EncoderStack(params)
    o_dense = tf.keras.layers.Dense(self.params["feature_dim"], activation="relu", name="dense_output")
    self.output_dense = DenseLayerWrapper(o_dense, params)
    self.general_mask_id = tf.constant([0])

    if params["category_embedding"]:
        self.category_embedding = tf.keras.layers.Embedding(params["categories_count"], output_dim=self.params["feature_dim"])


  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, inputs, training):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: input tensor list of size 1 or 2.
        First item, inputs: float tensor with shape [batch_size, input_length, feature_dim].
        Second item (optional), targets: None or float tensor with shape
          [batch_size, target_length, feature_dim].
      training: boolean, whether in training mode or not.

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, feature_dim]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          outputs: [batch_size, feature_dim]
          scores: [batch_size, float]}
      Even when float16 is used, the output tensor(s) are always float32.

    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """

    inputs, categories, mask_positions = inputs[0], inputs[1], inputs[2]

    if self.params["masking_mode"] == "single-token":
        mask_tensor = self.tokens_embedding(self.general_mask_id)
        # Repeat the mask tensor to match the count of masked items
        mask_tensors = tf.repeat(mask_tensor, tf.shape(mask_positions)[0])
        # Reshape to (number of masked items, feature_dim)
        mask_tensors = tf.reshape(mask_tensors, shape=(-1, self.params["feature_dim"]))
        r = tf.range(0, limit=tf.shape(mask_positions)[0], dtype="int32")
        r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])
        indices = tf.concat([r, mask_positions], axis=-1)
        indices = tf.squeeze(indices, axis=[1])
        inputs = tf.tensor_scatter_nd_update(inputs, indices, mask_tensors)

    if self.params["category_embedding"]:
        flat_categories = tf.reshape(categories, shape=(-1, 1))

        # Compute padding mask # TODO: Make function out of this
        unpacked_categories = tf.reshape(categories, shape=[-1])
        unpacked_length = tf.shape(unpacked_categories)[0]
        padding_mask = tf.equal(unpacked_categories, 0)
        padding_mask = tf.math.logical_not(padding_mask)
        padding_mask = tf.cast(padding_mask, dtype="float32")
        mask_matrix = tf.zeros(shape=(unpacked_length, unpacked_length))
        mask_matrix = tf.linalg.set_diag(mask_matrix, padding_mask)

        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        embedded_categories = self.category_embedding(flat_categories)
        embedded_categories = tf.einsum("ij,jk->ik", mask_matrix, embedded_categories)
        embedded_categories = tf.reshape(embedded_categories, shape=(batch_size, seq_length, self.params["feature_dim"]))
        inputs = tf.keras.layers.add([inputs, embedded_categories])


    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("Transformer"):
      encoder_inputs = self.input_dense(inputs)
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      reduced_inputs = tf.equal(encoder_inputs, 0)
      reduced_inputs = tf.reduce_all(reduced_inputs, axis=2)
      attention_bias = model_utils.get_padding_bias(reduced_inputs, True)


      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs = self.encode(encoder_inputs, attention_bias, training)
      output = self.output_dense(encoder_outputs)
      return output

  def encode(self, inputs, attention_bias, training):
    """Generate continuous representation for inputs.

    Args:
      inputs: float tensor with shape [batch_size, input_length, feature_dim].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      # TODO: Change embedding
      # embedded_inputs = self.embedding_softmax_layer(inputs)
      # embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])

      inputs_padding = model_utils.get_padding(inputs)
      attention_bias = tf.cast(attention_bias, self.params["dtype"])

      # with tf.name_scope("add_pos_encoding"):
      #   length = tf.shape(embedded_inputs)[1]
      #
      #   # TODO: Remove / Change positional encoding
      #   pos_encoding = model_utils.get_position_encoding(
      #       length, self.params["hidden_size"])
      #   pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
      #   encoder_inputs = embedded_inputs + pos_encoding

      encoder_inputs = inputs

      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params["layer_postprocess_dropout"])

      return self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training)


class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params["layer_postprocess_dropout"]

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    return x + y


class DenseLayerWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params):
    super(DenseLayerWrapper, self).__init__()
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params["layer_postprocess_dropout"]

  def build(self, input_shape):
    # Create normalization layer
    super(DenseLayerWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    # Get layer output
    y = self.layer(x, *args, **kwargs)

    # Postprocessing: apply dropout
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    return y


class EncoderStack(tf.keras.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params):
    super(EncoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    """Builds the encoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])

    # Create final layer normalization layer.
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(EncoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, encoder_inputs, attention_bias, inputs_padding, training):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
        1, input_length]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          encoder_inputs = self_attention_layer(
              encoder_inputs, attention_bias, training=training)
        with tf.name_scope("ffn"):
          encoder_inputs = feed_forward_network(
              encoder_inputs, training=training)

    return self.output_normalization(encoder_inputs)