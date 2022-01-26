import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.multi_head_attention import MultiHeadAttention
from tensorflow.python.ops.gen_math_ops import Mul
from MHA import *


class DecoderLayer(layers.Layer):
    def __init__(self, dims, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = MultiHeadAttention(dims, num_heads)
        self.mha2 = MultiHeadAttention(dims, num_heads)

        self.ffn = pointwise_feedforward(dims, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, mask, look_ahead_mask, padding_mask):

        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # add and norm

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # add and norm, (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(layers.Layer):
    def __init__(self, num_layers, dims, num_heads, dff):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.dims = dims
        self.num_heads = num_heads
        self.dff = dff

        self.layers = [DecoderLayer(dims, num_heads, dff)
                       for i in range(num_layers)]

    def call(self, x, enc_output, training, mask, look_ahead_mask, padding_mask):
        for layer in self.layers:
            output = layer(x, enc_output, training, mask,
                           look_ahead_mask, padding_mask)

        return output
