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

"""Defines the Fashion Outfit Encoder model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.transformer.model import model_utils
from official.transformer.v2 import ffn_layer
import src.models.encoder.layers as layers
import src.models.encoder.utils as utils


def create_model(params, is_train):
    """Creates a Fashion Encoder model."""
    with tf.name_scope("model"):

        categories = tf.keras.layers.Input((None,), dtype="int32", name="categories")
        mask_positions = tf.keras.layers.Input((None, 1), dtype="int32", name="mask_positions")

        if params["with_cnn"]:
            inputs = tf.keras.layers.Input((None, 299, 299, 3), dtype="float32", name="inputs")
        else:
            inputs = tf.keras.layers.Input((None, params["feature_dim"]), dtype="float32", name="inputs")

        preprocessor = FashionPreprocessorV2(params, "preprocessor")

        encoder_inputs, training_targets = preprocessor([inputs, categories, mask_positions], training=is_train)

        internal_model = FashionEncoder(params, name="encoder")
        ret = internal_model([encoder_inputs, categories], training=is_train)

        return tf.keras.Model([inputs, categories, mask_positions], [ret, training_targets])


class FashionPreprocessorV2(tf.keras.Model):

    def __init__(self, params, name=None):
        """Initialize layers to build preprocessor.

        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          name: name of the model.
        """
        super(FashionPreprocessorV2, self).__init__(name=name)
        self.params = params

        if params["with_cnn"]:
            self.cnn_extractor = CNNExtractor(params, "cnn_extractor")

        if self.params["masking_mode"] == "single-token":
            self.masking_layer = layers.SingleMasking(params)
        elif self.params["masking_mode"] == "category-masking":
            self.masking_layer = layers.CategoryMasking(params)

        dense_size = params["hidden_size"]

        if "category_embedding" in params:
            if params["category_merge"] == "add":
                self.category_embedding = layers.CategoryAdder(params)
            elif params["category_merge"] == "multiply":
                self.category_embedding = layers.CategoryMultiplier(params)
            elif params["category_merge"] == "concat":
                self.category_embedding = layers.CategoryConcater(params)
                dense_size = params["hidden_size"] - params["category_dim"]

        i_dense = tf.keras.layers.Dense(dense_size, activation=lambda x: tf.nn.leaky_relu(x),
                                        input_shape=(None, None, self.params["feature_dim"]), name="dense_input",
                                        activity_regularizer=tf.keras.regularizers.l2(self.params["dense_regularization"])
                                        )

        self.input_dense = DenseLayerWrapper(i_dense, params)

    def _add_category_embedding(self, inputs, categories, mask_positions):
        return self.category_embedding([inputs, categories, mask_positions])

    def call(self, inputs, *args, **kwargs):
        """Calculate target logits or inferred target sequences.

        Args:
          inputs: input tensor list of size 3
            First item, inputs: float tensor
             - When using already extracted features with shape [batch_size, seq_length, feature_dim]
             - When using images with shape [batch_size, input_length, image_width, image_height, 3]
            Second item, categories: int tensor with shape [batch_size, seq_length, feature_dim].
            Third item, mask positions: int tensor with shape [batch_size, seq_length, 1]
          training: boolean, whether in training mode or not.

        Returns:
          Embedded outfits with shape [batch_size, seq_length, feature_dim]

        Raises:
          NotImplementedError: If try to use padded decode method on CPU/GPUs.
        """

        inputs, categories, mask_positions = inputs[0], inputs[1], inputs[2]
        logger = tf.get_logger()
        training = kwargs["training"]
        if self.params["mode"] == "debug":
            logger.debug("Categories")
            logger.debug(categories)

        if self.params["with_cnn"]:
            # Extract Image features
            inputs = self.cnn_extractor([inputs, categories, mask_positions])

        training_targets = self.input_dense(inputs, training=training)
        masked_inputs = training_targets

        # Place mask tokens
        if mask_positions is not None:
            masked_inputs = self.masking_layer([masked_inputs, categories, mask_positions], training=training)
            if self.params["mode"] == "debug":
                logger.debug("Masked Inputs")
                logger.debug(masked_inputs)

        # Merge image features with category embedding
        if self.params["category_embedding"]:
            training_targets = self._add_category_embedding(training_targets, categories, None)
            masked_inputs = self._add_category_embedding(masked_inputs, categories, mask_positions)

            if self.params["mode"] == "debug":
                logger.debug("Masked inputs with categories")
                logger.debug(masked_inputs)

        return masked_inputs, training_targets


class CNNExtractor(tf.keras.Model):

    def __init__(self, params, name=None):
        """Initialize layers to build CNN extractor.

        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          name: name of the model.
        """
        super(CNNExtractor, self).__init__(name=name)
        self.params = params
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
        TODO: Documement
        Args:
            inputs: tuple (inputs, categories, mask_positions)
              inputs - tensor of shape (batch_size, seq_length, 299, 299, 3)
            training: T
        """
        logger = tf.get_logger()

        inputs, categories, mask_positions = inputs[0], inputs[1], inputs[2]
        if self.params["mode"] == "debug":
            logger.debug(inputs)

        mask_matrix = utils.compute_padding_mask_from_categories(categories)

        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        inputs = tf.reshape(inputs, shape=(-1, 299, 299, 3))
        cnn_outputs = self.cnn_model(inputs)
        cnn_outputs = tf.einsum("ij,jk->ik", mask_matrix, cnn_outputs)

        if self.params["mode"] == "debug":
            logger.debug(cnn_outputs)
        return tf.reshape(cnn_outputs, shape=(batch_size, seq_length, self.params["feature_dim"]))


class FashionEncoder(tf.keras.Model):
    """Encoder model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, params, name=None):
        """Initialize layers to build Encoder model.

        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          name: name of the model.
        """
        super(FashionEncoder, self).__init__(name=name)
        self.params = params

        self.encoder_stack = EncoderStack(params)

        if self.params["category_merge"] == "concat":
            units = self.params["feature_dim"]  # + self.params["category_dim"]
        else:
            units = self.params["feature_dim"]

        self.activity_regularizer = tf.keras.regularizers.l2(self.params["enc_regularization"])
        # o_dense = tf.keras.layers.Dense(units, activation=lambda x: tf.nn.leaky_relu(x),
        #                                 name="dense_output")
        # self.output_dense = DenseLayerWrapper(o_dense, params)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, inputs, training):
        """Calculate target logits or inferred target sequences.

        Args:
          inputs: input tensor list of size 2.
            First item, inputs: float tensor with shape [batch_size, seq_length, feature_dim].
            Second item, categories: None or float tensor with shape [batch_size, seq_length, 1].
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

        inputs, categories = inputs[0], inputs[1]

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            logger = tf.get_logger()
            if self.params["mode"] == "debug":
                logger.debug("Transformer inputs")
                logger.debug(inputs)

            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            # reduced_inputs = tf.equal(categories, 0)   #TODO: Is this needed?
            #reduced_inputs = tf.reduce_all(reduced_inputs, axis=2)

            attention_bias = model_utils.get_padding_bias(categories, 0)

            # TODO: Code for keys from categories
            one_hot_categories = tf.one_hot(categories, self.params["categories_count"])

            if self.params["mode"] == "debug":
                logger.debug("Categories")
                logger.debug(categories)
                logger.debug("One hot")
                logger.debug(one_hot_categories)

            output = self.encode(inputs, categories, attention_bias, training)

            self.add_loss(self.activity_regularizer(output))

            # if self.params["hidden_size"] != self.params["feature_dim"]:
            #     output = self.output_dense(output, training=training)

            return output

    def encode(self, inputs, categories, attention_bias, training):
        """Generate continuous representation for inputs.

        Args:
          inputs: float tensor with shape [batch_size, input_length, feature_dim].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
          training: boolean, whether in training mode or not.

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            attention_bias = tf.cast(attention_bias, self.params["dtype"])
            encoder_inputs = inputs

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.params["layer_postprocess_dropout"])

            return self.encoder_stack(
                encoder_inputs, categories, attention_bias, training=training)


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
    """Wrapper class that applies layer dropout."""

    def __init__(self, layer, params):
        super(DenseLayerWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params["i_dense_dropout"]

    def build(self, input_shape):
        super(DenseLayerWrapper, self).build(input_shape)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        training = kwargs["training"]
        logger = tf.get_logger()

        # Get layer output
        y = self.layer(x, *args, **kwargs)
        if self.params["mode"] == "debug":
            logger.debug("Before Dropout")
            logger.debug(y)
        # Postprocessing: apply dropout
        if training:

            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
            if self.params["mode"] == "debug":
                logger.debug("Dense dropout applied")
                logger.debug("After Dropout")
                logger.debug(y)
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
            self_attention_layer = layers.SelfAttention(
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

    def call(self, encoder_inputs, categories, attention_bias, training):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
            1, input_length]
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
                        encoder_inputs, None, attention_bias, training=training)  # TODO: Categories set to None
                with tf.name_scope("ffn"):
                    encoder_inputs = feed_forward_network(
                        encoder_inputs, training=training)

        return self.output_normalization(encoder_inputs)
