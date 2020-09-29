import tensorflow as tf
import numpy as np


class AutoEncoder(tf.keras.Model):
    def __init__(self,layers_scheme,embedding_size,seed=None):
        super(AutoEncoder,self).__init__()
        self.layers_scheme = layers_scheme
        self.embedding_size = embedding_size
        self.seed = seed

    def build(self, input_shape):
        kernel_init = tf.keras.initializers.GlorotNormal(self.seed)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(n, activation=act, kernel_initializer=kernel_init)
            for n, act in self.layers_scheme
        ]+ [
            tf.keras.layers.Dense(self.embedding_size, kernel_initializer=kernel_init)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(n, activation=act, kernel_initializer=kernel_init)
            for n, act in self.layers_scheme[::-1]
        ]+ [
            tf.keras.layers.Dense(input_shape[1], kernel_initializer=kernel_init)
        ])

    def encode(self,input,trainitg=None):
        return self._encode(np.atleast_2d(input).astype(np.float32),trainitg)

    def decode(self,input,trainitg=None):
        return self._decode(np.atleast_2d(input).astype(np.float32),trainitg)

    @tf.function
    def _encode(self,input,trainig):
        return self.encoder(input,trainig)

    @tf.function
    def _decode(self, input, trainig):
        return self.decoder(input, trainig)

    @tf.function
    def call(self, inputs, training=None):
        rep = self._encode(inputs,training)
        dec =  self._decode(rep,training)
        return dec, rep

