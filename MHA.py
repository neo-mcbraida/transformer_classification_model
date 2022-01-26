import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def pointwise_feedforward(dims, dff):
    return keras.Sequential([
        layers.Dense(dff, activation="relu"),
        layers.Dense(dims)
    ])

# class MultiHeadAttention(layers.Layer):
#     def __init__(self, dims, num_heads):
#         super(MultiHeadAttention, self).__init__()

#         self.dims = dims
#         self.num_heads = num_heads

#         # tensorflow says:
#         # Instead of one single attention head, Q, K, and V are split
#         # into multiple heads because it allows the model to jointly
#         # attend to information from different representation subspaces
#         # at different positions. After the split each head has a reduced
#         # dimensionality, so the total computation cost is the same as a single
#         # head attention with full dimensionality.
#         assert dims % num_heads == 0
#         self.depth = dims // num_heads # number of dims in each head

#         self.wq = layers.Dense(dims)
#         self.wk = layers.Dense(dims)
#         self.wv = layers.Dense(dims)

#         self.dense = layers.Dense(dims)

#     def split_heads(self, x, batch_size):
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#         return x.transpose(x, perm=[0, 2, 1, 3])

#     def call(self, q, k, v, mask):
#         batch_size = tf.shape(q)[0]

#         q = self.wq(q)
#         k = self.wk(k)
#         v = self.wv(v)

#         q = self.split_heads(q, batch_size)
#         k = self.split_heads(k, batch_size)
#         v = self.split_heads(v, batch_size)

#         scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

#         scaled_attention = scaled_attention.transpose(scaled_attention, (0, 2, 3, 1))


#         concat_attention = tf.reshape(scaled_attention,
#                                     (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

#         output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

#         return output
