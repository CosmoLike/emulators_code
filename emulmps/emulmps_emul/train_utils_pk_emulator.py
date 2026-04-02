# Essential components required for fastMPS emulator


import numpy as np
import tensorflow as tf
from keras import layers

#k=1e-5 to 1e2 (with 2400 steps in log space)
ks = np.logspace(-5, 2, 2400)
z1_mps = np.linspace(0, 2, 100, endpoint=False)
z2_mps = np.linspace(2, 10, 10, endpoint=False)
z3_mps = np.linspace(10, 50, 12)
z_mps = np.concatenate((z1_mps, z2_mps, z3_mps), axis=0)

        

# Bernardo's code for KAN model in PyTorch
class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return (X* self.std) + self.mean
    

class CustomActivationLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        # IMPORTANT: Always call the super constructor first
        super(CustomActivationLayer, self).__init__(**kwargs)
        self.units = units
        # Set a dictionary property for compatibility with get_config
        self.input_spec = layers.InputSpec(min_ndim=2)

    def build(self, input_shape):
        # Create trainable weights
        self.beta = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="beta")
        self.gamma = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="gamma")
        super(CustomActivationLayer, self).build(input_shape)

    def call(self, x):
        # Activation function logic
        func = tf.add(self.gamma, tf.multiply(tf.sigmoid(tf.multiply(self.beta, x)), tf.subtract(1.0, self.gamma)))
        return tf.multiply(func, x)
    
    def get_config(self):
        # Get the base config from the Layer class (includes name, dtype, etc.)
        config = super(CustomActivationLayer, self).get_config()
        # Add the required constructor arguments
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        # Use the saved config to reconstruct the class instance
        return cls(**config)
    # ========================================

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
