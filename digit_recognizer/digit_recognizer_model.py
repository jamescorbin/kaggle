import sys
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

_height, _width = 28, 28

def load_train_data(fn: str):
    df = pd.read_csv(fn, header=0)
    x_train = (df.drop("label", axis=1)
               .values
               .reshape((len(df), _height, _width, 1))
               .astype(np.float32))
    y_train = pd.DataFrame(df[["label"]]).values
    return x_train, y_train

def load_test_data(fn: str):
    df = pd.read_csv(fn, header=0)
    x_train = (df
               .values
               .reshape((len(df), _height, _width, 1))
               .astype(np.float32))
    return x_train

def convert_gray_to_float(images: np.array) -> np.array:
    images = (images -  128) / 255
    return images

def make_tf_dataset(x_train: np.array,
                    y_train: np.array,
                    shards: int=10,
                    validation_k: int=3,
                    batch_size: int=128) -> tf.data.Dataset:
    x = tf.data.Dataset.from_tensor_slices(x_train)
    y = tf.data.Dataset.from_tensor_slices(y_train)
    xy = tf.data.Dataset.zip((x, y))
    ds_train = (xy.enumerate()
                .filter(lambda x, y: x % shards > validation_k)
                .map(lambda x, y: y))
    ds_valid = (xy.enumerate()
                .filter(lambda x, y: x % shards <= validation_k)
                .map(lambda x, y: y))
    ds_train = ds_train.batch(batch_size)
    ds_valid = ds_valid.batch(batch_size)
    return ds_train, ds_valid

def build_model(conv2d_dim: int,
                conv2d_kernel_size: Tuple[int, int]=(_height, _width),
                pool_size: Tuple[int, int]=(_height, _width),
                dense0_dim: int=40,
                dense1_dim: int=10,
                ) -> tf.keras.Model:
    img_inputs = tf.keras.Input(shape=(_height, _width, 1))
    conv2d_layer = tf.keras.layers.Conv2D(filters=conv2d_dim,
                                   kernel_size=conv2d_kernel_size,
                                   activation="relu",
                                   use_bias=True,)
    pool2d_layer = tf.keras.layers.AveragePooling2D(
                                    pool_size=pool_size,)
    flat_layer = tf.keras.layers.Flatten()
    dense0_layer = tf.keras.layers.Dense(dense0_dim,
                                         activation="relu")
    dense1_layer = tf.keras.layers.Dense(dense1_dim,
                                         activation="sigmoid")
    #loss = tf.keras.losses.CategoricalCrossentropy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    opt = tf.keras.optimizers.RMSprop()
    x = conv2d_layer(img_inputs)
    x = pool2d_layer(x)
    x = flat_layer(x)
    x = dense0_layer(x)
    x = dense1_layer(x)
    model = tf.keras.Model(inputs=[img_inputs],
                           outputs=x,
                           name="digit_model")
    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=["accuracy"],)
    return model

def train_model(model: tf.keras.Model,
                ds_train: tf.data.Dataset,
                ds_valid: tf.data.Dataset,
                ):
    model.fit(
            ds_train,
            validation_data=ds_valid,
            shuffle=True,
            epochs=10)

def decode_predictions(predictions: np.array) -> np.array:
    return np.argmax(predictions, axis=1).flatten()

if __name__=="__main__":
    tf.random.set_seed(50)
    train_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "train.csv"))
    x_train, y_train = load_train_data(train_fn)
    test_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "test.csv"))
    x_test = load_test_data(test_fn)
    x_train = convert_gray_to_float(x_train)
    x_test = convert_gray_to_float(x_test)
    model = build_model(20, (5, 5), (3, 3))
    ds_train, ds_valid = make_tf_dataset(x_train, y_train)
    train_model(model, ds_train, ds_valid)
    y_pred = model.predict(x_test)
    y_pred = decode_predictions(y_pred)
    results = pd.DataFrame(
        {"ImageId": np.arange(1, y_pred.shape[0] + 1),
               "Label": y_pred.flatten()})
    results.to_csv("results.csv", index=False)
