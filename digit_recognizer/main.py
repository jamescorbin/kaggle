import sys
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

_height, _width = 28, 28

def load_train_data(fn: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(fn, header=0)
    x_train = (df.drop("label", axis=1)
               .values
               .reshape((len(df), _height, _width, 1))
               .astype(np.float32))
    y_train = pd.DataFrame(df[["label"]]).values
    return x_train, y_train

def load_test_data(fn: str) -> pd.DataFrame:
    df = pd.read_csv(fn, header=0)
    x_test = (df
               .values
               .reshape((len(df), _height, _width, 1))
               .astype(np.float32))
    return x_test

def convert_gray_to_float(images: np.array) -> np.array:
    images = (images -  128) / 255.0
    return images

def decode_predictions(predictions: np.array) -> np.array:
    return np.argmax(predictions, axis=1).flatten()

if __name__=="__main__":
    config_fn = "config-model.json"
    with open(config_fn, "r") as f:
        config = json.load(f)
    tf.random.set_seed(config["seed"])
    train_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "train.csv"))
    x_train, y_train = load_train_data(train_fn)
    test_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "test.csv"))
    x_train = convert_gray_to_float(x_train)
    x_pred = load_test_data(test_fn)
    x_pred = convert_gray_to_float(x_pred)
    config = {}
    strategy = tf.distribute.get_strategy()
    batch_size = config["batch_size"] * strategy.num_replicas_in_sync
    with strategy.scope():
        x_ds = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(x_train),
            tf.data.Dataset.from_tensor_slices(y_train)))
        x_train = (x_ds
                .enumerate()
                .filter(lambda x, y:
                        x % config["split_mod_k"] <= config["train_k"])
                .map(lambda x, y: y)
                .shuffle(config["shuffle"])
        x_valid = (x_ds
                .enumerate()
                .filter(lambda x, y:
                        (x % config["split_mod_k"] > config["train_k"])
                        & (x % config["split_mod_k"] <= config["valid_k"]))
                .map(lambda x, y: y))
        x_test = (x_ds
                .enumerate()
                .filter(lambda x, y:
                        x % config["split_mod_k"] > config["valid_k"])
                .map(lambda x, y: y))
        tfboard = tf.keras.callbacks.TensorBoard(
                log_dir=tfboard_log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=False,
                update_freq="epoch",
                profile_batch=0,
                embeddings_freq=1,
                embeddings_metadata=None,)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                model_save_pt,
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                save_freq="epoch",
                options=None,
                initial_value_threshold=None,)
        early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=0,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True)
        callbacks = [tfboard, model_checkpoint, early_stopping]
        model = build_model(config)
        hist = model.fit(
                x_train
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE),
                validation_data=x_valid.batch(batch_size),
                epochs=config["epochs"],
                callbacks=callbacks)
        model.evaluate(
                x_train.batch(batch_size),
                callbacks=callbacks)
        model.evaluate(
                x_valid.batch(batch_size),
                callbacks=callbacks)
        model.evaluate(
                x_test.batch(batch_size),
                callbacks=callbacks)
        y_pred = model.predict(x_test)
    y_pred = decode_predictions(y_pred)
    results = pd.DataFrame(
        {"ImageId": np.arange(1, y_pred.shape[0] + 1),
               "Label": y_pred.flatten()})
    results.to_csv("results.csv", index=False)