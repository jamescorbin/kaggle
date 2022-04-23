"""
kaggle competitions download -c tpu-getting-started
"""
import sys
import os
import math
import json
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import tensorflow_datasets as tfds
import imageaug

TWO_FIVE_FIVE = tf.constant(
        255.0,
        shape=[],
        dtype=tf.float32,
        name="255")

@tf.function
def transform(x: tf.Tensor) -> tf.Tensor:
    return imageaug.apply_affine_transform(
            x,
            imageaug.get_random_transformation())

def get_strategy() -> Tuple[Optional["tpu"], Optional["strategy"]]:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    return tpu, strategy

def read_tfrecord(example,
                  dim: int=192,
                  labeled: bool=True,
                  ):
    tfrec_format = {"image": tf.io.FixedLenFeature([], tf.string)}
    tfrec_format["id"] = tf.io.FixedLenFeature([], tf.string)
    if labeled:
        tfrec_format["class"] = tf.io.FixedLenFeature([], tf.int64)
    parsed = tf.io.parse_single_example(example, tfrec_format)
    image = cast_image_float(
            tf.reshape(
                tf.image.decode_jpeg(parsed["image"], channels=3),
                (dim, dim, 3)))
    out = {"image": image}
    out["id"] = tf.io.decode_raw(parsed["id"],
                                 out_type=tf.uint8,
                                 fixed_length=9)
    if labeled:
        out["class"] = tf.cast(parsed["class"], tf.int64)
    return out

@tf.function
def cast_image_float(image: tf.Tensor) -> tf.Tensor:
    image = tf.divide(
                tf.cast(image, tf.float32),
                TWO_FIVE_FIVE)
    return image

def get_training_dataset(
        config: Dict[str, Any],
        top_dir: str=os.path.join(os.getcwd(), "data"),
        ) -> tf.data.Dataset:
    if not os.path.exists(top_dir):
        from kaggle_datasets import KaggleDatasets
        top_dir=KaggleDatasets().get_gcs_path("tpu-getting-started")
    dirname = lambda x: f"tfrecords-jpeg-{x}x{x}"
    tf_recs = tf.io.gfile.glob(f"{top_dir}/"
                               f"{dirname(dim)}/"
                               "(train|valid)/"
                               "*.tfrec")
    read_f = lambda x: read_tfrecord(x,
            dim=dim,
            labeled=True,)
    x_ds = tf.data.TFRecordDataset(tf_recs).map(read_f)
    x_ds = tf.data.Dataset.zip((
        x_ds.map(lambda x: {"image": x["image"], "id": x["id"]}),
        x_ds.map(lambda x: {"class": x["class"]})))
    x_train = (x_ds
            .enumerate()
            .filter(lambda x, y:
                    x % config["split_mod_k"] <= config["train_k"])
            .map(lambda x, y: y))
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
    return x_train, x_valid, x_test

def augment_train(x_train: tf.data.Dataset,
                  batch_size: int,
                  config) -> tf.data.Dataset:
    x_train = x_train.repeat(config["repeat"])
    x_train = x_train.shuffle(config["shuffle"])
    x_train = x_train.batch(batch_size, drop_remainder=True)
    x_train.apply(lambda x, y:
                  x: {"image": transform(x["image"]),
                      "id": x["id"]},
                  y)
    return x_train

def get_prediction_dataset(
        config: Dict[str, Any],
        top_dir: str=os.path.join(os.getcwd(), "data"),
        ) -> tf.data.Dataset:
    if not os.path.exists(top_dir):
        from kaggle_datasets import KaggleDatasets
        top_dir=KaggleDatasets().get_gcs_path("tpu-getting-started")
    dirname = lambda x: f"tfrecords-jpeg-{x}x{x}"
    tf_recs = tf.io.gfile.glob(f"{top_dir}/"
                               f"{dirname(dim)}/"
                               f"{test}/*.tfrec")
    read_f = lambda x: read_tfrecord(x,
            dim=dim,
            labeled=False,)
    ds = tf.data.TFRecordDataset(tf_recs).map(read_f)
    return ds

def test_image_transform() -> None:
    import matplotlib.pyplot as plt
    dims = [192, 224, 331, 512]
    dim = dims[-1]
    ds = get_dataset(dim,
                    repeat=None,
                    buffer_size=None,
                    batch_size=None,
                    apply_transformation=False,
                    train_test="train",)
    ds = ds.map(lambda x, y: x)
    ds = ds.map(lambda x: x["image"])
    ds = ds.skip(100).take(1).repeat(2)
    ds = tfds.as_numpy(ds)
    it = iter(ds)
    ds0 = next(it)
    ds1 = transform(next(it))
    fix, ax = plt.subplots(1, 2)
    ax[0].imshow(ds0)
    ax[1].imshow(ds1)
    plt.savefig("test_image.png")

def main(
        batch_size: int=256,
        repeat: int=3,
        buffer_size: int=10):
    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    config_fn = "./cofing-model.json"
    with open(config_fn, "r") as f:
        config = json.load(f)
    print(f"TF version: {tf.__version__}")
    tf.random.set_seed(config["seed"])
    tpu, strategy = get_strategy()
    on_tpu = True if tpu else False
    print(f"TPU: {tpu}")
    print(f"Strategy: {strategy}")
    dim = config["image_shape"][0]
    print(f"Image dimension: {dim}")
    print(f"Epochs initial: {config['epochs_init']}")
    print(f"Epochs tuning: {config['epochs_tune']}")
    batch_size = config["batch_size"] * strategy.num_replicas_in_sync
    with strategy.scope():
        x_train, x_valid, x_test = get_training_dataset(config)
        x_train = augment_train(x_train, batch_size, config=config)
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
        model = FlowerModel(
                config=config,
                name="flower_model")
        print("Initial training: DenseNet not trainable")
        hist0 = model.fit(
                ds_train,
                validation_data=x_valid.batch(batch_size),
                callbacks=callbacks,
                epochs=config["epochs_init"])
        print("Fine-tuning training: DenseNet trainable")
        model.set_trainable_recompile()
        hist1 = model.fit(
                ds_train,
                validation_data=ds_valid.batch(batch_size),
                epochs=config["epochs_tune"],
                callbacks=[early_stopping])
        model.evaluate(
                x_train.batch(batch_size),
                callbacks=callbacks)
        model.evaluate(
                x_valid.batch(batch_size),
                callbacks=callbacks)
        model.evaluate(
                x_test.batch(batch_size),
                callbacks=callbacks)
    ds_test = get_dataset(
                    dim,
                    repeat=None,
                    buffer_size=None,
                    batch_size=64,
                    apply_transformation=False,
                    train_test="test",)
    prediction = model.predict(ds_test)
    prediction.pop("class")
    ids = np.array(["".join([chr(x) for x in pred])
                    for pred in prediction["id"]])
    prediction["id"] = ids
    prediction = pd.DataFrame(prediction)
    return model, (hist0, hist1), prediction

if __name__=="__main__":
    model, history, prediction = main(
            batch_size=64,
            epochs_init=2,
            epochs_tune=5,
            repeat=3,
            buffer_size=10)
