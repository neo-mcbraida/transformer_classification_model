import tensorflow as tf
import numpy as np
import encoder
import decoder
import MHA
import random


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class Transformer(tf.keras.Model):
    def __init__(self, dims, num_heads, dff, enc_layers, dec_layers, out_dims, rate=0.1):
        super().__init__()
        #self.flattener = tf.keras.layers.Dense(input_shape=(3,25, 25))
        self.encoder = encoder.Encoder(enc_layers, dims, dff, rate)
        self.decoder = decoder.Decoder(dec_layers, dims, num_heads, dff)
        self.final_layer = MHA.pointwise_feedforward(dff, out_dims)

    def call(self, inputs, targets, training=False):
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            inputs, targets)

        #flattened_input = self.flattener(inputs)

        encodings = self.encoder(inputs, training, enc_padding_mask)

        decodings = self.decoder(
            encodings, encodings, training, look_ahead_mask, dec_padding_mask)

        output = self.final_layer(decodings)

        return output

    # def train(self, batch_size, epochs, inputs, outputs):
    #     """
    #     inputs is [batches, channels, height, width]
    #     """
    #     num_batches = len(inputs) // batch_size
    #     for epoch in epochs:
    #         both = list(zip(inputs, outputs))
    #         random.shuffle(both)
    #         for i in range(num_batches):
    #             base = i * batch_size
    #             inputs, outputs = zip(*(both[base:base+batch_size]))
    #             #inp_batch = inputs[base:base + batch_size]
    #             with tf.GradientTape as tape:
    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
