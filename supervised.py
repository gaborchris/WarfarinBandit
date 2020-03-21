import tensorflow as tf
import numpy as np

def get_model(feature_dim):
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(32, input_shape=(feature_dim,), activation='relu'),
        tf.keras.layers.Dense(1, input_shape=(feature_dim,), activation=None)
    ])
    model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.mean_squared_error)
    return model




