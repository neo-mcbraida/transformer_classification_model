import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import train
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras.layers.multi_head_attention import MultiHeadAttention
from tensorflow.python.keras.utils import layer_utils
from MHA import *

class EncoderLayer(layers.Layer):
    def __init__(self, dims, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.MHA = MultiHeadAttention(dims, num_heads)
        self.ffn = pointwise_feedforward(dims, dff)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self.layers_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layers_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask):
        attn_output = self.MHA(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layers_norm1(attn_output + x)

        out2 = self.dff(out1)
        out2 = self.dropout2(out2, training=training)

        output = self.layers_norm2(out1 + out2)

        return output

class Encoder(layers.Layer):
    def __init__(self, num_layers, dims, dff, num_heads):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.dims = dims
        self.dff = dff
        self.num_heads = num_heads

        self.layers = [EncoderLayer(self.dims, self.num_heads, self.dff)
                        for i in range(self.num_layers)]

    def call(self, x, training, mask):
        # add positional encoding here

        for layer in self.layers:
            output = layer(x, training, mask)
        
        return output