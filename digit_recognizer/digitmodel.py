import sys
import os
from typing import Tuple, Dict, Any
import tensorflow as tf

def build_model(
            config: Dict[str, Any],
            output_dim: int=10,
            ) -> tf.keras.Model:
    img_inputs = tf.keras.Input(shape=config["image_shape"])
    conv2d_layer = tf.keras.layers.Conv2D(
            filters=config["conv2d_dim"],
            kernel_size=config["conv2d_kernel"],
            activation="relu",
            use_bias=True,
            name="conv2d_layer")
    pool2d_layer = tf.keras.layers.AveragePooling2D(
            pool_size=config["pool2d_size"],
            name="pooling_layer")
    flat_layer = tf.keras.layers.Flatten()
    dense0_layer = tf.keras.layers.Dense(
            config["dense_dim_0"],
            activation="relu")
    dense1_layer = tf.keras.layers.Dense(output_dim,
            activation="sigmoid")
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    opt = tf.keras.optimizers.RMSprop(
            learning_rate=config["learning_rate"])
    x = conv2d_layer(img_inputs)
    x = pool2d_layer(x)
    x = flat_layer(x)
    x = dense0_layer(x)
    x = dense1_layer(x)
    model = tf.keras.Model(
            inputs=[img_inputs],
            outputs=[x],
            name="digit_model")
    model.compile(
            loss=loss,
            optimizer=opt,
            metrics=["accuracy"],)
    return model
