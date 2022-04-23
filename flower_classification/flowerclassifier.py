import sys
import os
from typing import Dict, Any, Tuple

class FlowerModel(tf.keras.Model):
    def __init__(self,
                 config: Dict[str, Any],
                 classes: int,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config = config
        from tensorflow.keras.applications import DenseNet201
        self.rnet = DenseNet201(
                input_shape=config["image_shape"],
                weights="imagenet",
                include_top=False,)
        #from tensorflow.keras.applications import Xception
        #self.rnet = Xception(
        #        input_shape=config["image_shape"],
        #        weights="imagenet",
        #        include_top=False,)
        self.rnet.trainable = False
        self.pooling = tf.keras.layers.GlobalAveragePooling2D(
                name="pooling")
        self.flat = tf.keras.layers.Flatten(
                name="flatten_pooling")
        self.dense_hidden = tf.keras.layers.Dense(
                units=config["hidden_dim_0"],
                activation="relu",
                name="dense_hidden")
        self.dropout = tf.keras.layers.Dropout(
                config["dropout_rate_0"],
                name="dropout_layer")
        self.out_layer = tf.keras.layers.Dense(
                config["classes"],
                activation="softmax",
                dtype=tf.float32,
                name="flower_class")
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        opt = tf.keras.optimizers.Adam(
                learning_rate=config["initial_learning_rate"])
        self.compile(
                optimizer=opt,
                loss={"class": loss,
                      "label": None,
                      "id": None},
                metrics={"class": [metric]})

    def set_trainable_recompile(self):
        self.rnet.trainable = True
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        opt = tf.keras.optimizers.Adam(
                learning_rate=self.config["final_learning_rate"])
        self.compile(
                optimizer=opt,
                loss={"class": loss,
                      "label": None,
                      "id": None},
                metrics={"class": [metric]})

    def call(self, inputs):
        x = inputs["image"]
        x = self.rnet(x)
        x = self.pooling(x)
        x = self.flat(x)
        x = self.dense_hidden(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        label = tf.reshape(
                tf.math.top_k(x, k=1).indices,
                shape=[-1])
        outputs = {"class": x,
                   "label": label,
                   "id": inputs["id"]}
        return outputs
