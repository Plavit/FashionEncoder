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
    if is_train:
        inputs = tf.keras.layers.Input((None, params["feature_dim"]), dtype="float32", name="inputs")
        categories = tf.keras.layers.Input((None,), dtype="int32", name="categories")
        mask_positions = tf.keras.layers.Input((None, 1), dtype="int32", name="mask_positions")
        internal_model = FashionEncoder(params, name="transformer_v2")
        ret = internal_model([inputs, categories, mask_positions], training=True)
        internal_model.summary()
        # outputs, scores = ret["outputs"], ret["scores"]
        return tf.keras.Model([inputs, categories, mask_positions], [ret])
    else:
        inputs = tf.keras.layers.Input((None, params["feature_dim"]), dtype="float32", name="inputs")
        categories = tf.keras.layers.Input((None,), dtype="int32", name="categories")
        mask_positions = tf.keras.layers.Input((None, 1), dtype="int32", name="mask_positions")
        internal_model = FashionEncoder(params, name="transformer_v2")
        ret = internal_model([inputs, categories, mask_positions], training=False)
        internal_model.summary()
        # outputs, scores = ret["outputs"], ret["scores"]
        return tf.keras.Model([inputs, categories, mask_positions], [ret])


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

    self.input_dense = tf.keras.layers.Dense(self.params["hidden_size"], activation="relu",
                                             input_shape=(None, None, self.params["feature_dim"]), name="dense_input")
    self.encoder_stack = EncoderStack(params)
    self.output_dense = tf.keras.layers.Dense(self.params["feature_dim"], activation="relu", name="dense_output")
    self.general_mask_id = tf.constant([0])

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
        mask_tensors = tf.repeat(mask_tensor, tf.shape(mask_positions)[0])
        mask_tensors = tf.reshape(mask_tensors, shape=(-1, self.params["feature_dim"]))
        r = tf.range(0, limit=tf.shape(mask_positions)[0], dtype="int32")
        r = tf.reshape(r, shape=[tf.shape(r)[0], -1, 1])
        indices = tf.squeeze(tf.concat([r, mask_positions], axis=-1))
        masked_inputs = tf.tensor_scatter_nd_update(inputs, indices, mask_tensors)

    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("Transformer"):
      encoder_inputs = self.input_dense(masked_inputs)
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

  def predict_2(self, encoder_outputs, attention_bias, training):
    """Generate logits for each value in the target sequence.

    Args:
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    decoder_inputs = tf.zeros((25, 8, self.params["hidden_size"]))
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      decoder_inputs = tf.cast(decoder_inputs, self.params["dtype"])
      attention_bias = tf.cast(attention_bias, self.params["dtype"])
      # with tf.name_scope("shift_targets"):
      #   # Shift targets to the right, and remove the last element
      #   decoder_inputs = tf.pad(decoder_inputs,
      #                           [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

      # with tf.name_scope("add_pos_encoding"):
      #   length = tf.shape(decoder_inputs)[1]
      #   pos_encoding = model_utils.get_position_encoding(
      #       length, self.params["hidden_size"])
      #   pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
      #   decoder_inputs += pos_encoding
      # TODO: Add positional encoding

      length = tf.shape(decoder_inputs)[1]

      if training:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, rate=self.params["layer_postprocess_dropout"])

      # Run values
      decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
          length, dtype=self.params["dtype"])

      outputs = self.decoder_stack(
          decoder_inputs,
          encoder_outputs,
          decoder_self_attention_bias,
          attention_bias,
          training=training)

      return outputs

  def _get_symbols_to_logits_fn(self, max_decode_length, training):
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = model_utils.get_position_encoding(
        max_decode_length + 1, self.params["hidden_size"])
    timing_signal = tf.cast(timing_signal, self.params["dtype"])
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length, dtype=self.params["dtype"])

    # TODO(b/139770046): Refactor code with better naming of i.
    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1].
        i: Loop index.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)

      if self.params["padded_decode"]:
        timing_signal_shape = timing_signal.shape.as_list()
        decoder_input += tf.slice(timing_signal, [i, 0],
                                  [1, timing_signal_shape[1]])

        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, i, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      else:
        decoder_input += timing_signal[i:i + 1]

        self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self.decoder_stack(
          decoder_input,
          cache.get("encoder_outputs"),
          self_attention_bias,
          cache.get("encoder_decoder_attention_bias"),
          training=training,
          cache=cache,
          decode_loop_step=i if self.params["padded_decode"] else None)
      logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, training):
    """Return predicted sequence."""
    encoder_outputs = tf.cast(encoder_outputs, self.params["dtype"])
    if self.params["padded_decode"]:
      batch_size = encoder_outputs.shape.as_list()[0]
      input_length = encoder_outputs.shape.as_list()[1]
    else:
      batch_size = tf.shape(encoder_outputs)[0]
      input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.params["extra_decode_length"]
    encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             self.params["dtype"])

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(
        max_decode_length, training)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    # pylint: disable=g-complex-comprehension
    init_decode_length = (
        max_decode_length if self.params["padded_decode"] else 0)
    num_heads = self.params["num_heads"]
    dim_per_head = self.params["hidden_size"] // num_heads
    cache = {
        "layer_%d" % layer: {
            "k":
                tf.zeros([
                    batch_size, init_decode_length, num_heads, dim_per_head
                ],
                         dtype=self.params["dtype"]),
            "v":
                tf.zeros([
                    batch_size, init_decode_length, num_heads, dim_per_head
                ],
                         dtype=self.params["dtype"])
        } for layer in range(self.params["num_hidden_layers"])
    }
    # pylint: enable=g-complex-comprehension

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=max_decode_length,
        eos_id=EOS_ID,
        padded_decode=self.params["padded_decode"],
        dtype=self.params["dtype"])

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}


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