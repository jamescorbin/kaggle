"""
kaggle competitions download -c tpu-getting-started
"""
import sys
import os
import math
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

def get_dataset(dim: int,
                repeat: Optional[int],
                buffer_size: Optional[int],
                batch_size: Optional[int],
                train_test: str="train",
                apply_transformation: bool=False,
                top_dir: str=os.path.join(os.getcwd(), "data"),
                ) -> tf.data.Dataset:
    if not os.path.exists(top_dir):
        from kaggle_datasets import KaggleDatasets
        top_dir=KaggleDatasets().get_gcs_path("tpu-getting-started")
    dirname = lambda x: f"tfrecords-jpeg-{x}x{x}"
    tf_recs = tf.io.gfile.glob(f"{top_dir}/"
                               f"{dirname(dim)}/"
                               f"{train_test}/*.tfrec")
    if train_test in ("train", "val"):
        read_f = lambda x: read_tfrecord(x,
                                         dim=dim,
                                         labeled=True,)
    else:
        read_f = lambda x: read_tfrecord(x,
                                         dim=dim,
                                         labeled=False,)
    ds = tf.data.TFRecordDataset(tf_recs).map(read_f)
    if repeat is not None:
        ds = ds.repeat(repeat)
    if train_test in ("train", "val"):
        if apply_transformation:
            xds = ds.map(lambda x: {
                "image": transform(x["image"]),
                "id": x["id"]})
        else:
            xds = ds.map(lambda x: {
                "image": x["image"],
                    "id": x["id"]})
        yds = ds.map(lambda x: {"class": x["class"]})
        ds = tf.data.Dataset.zip((xds, yds))
    if buffer_size is not None:
        ds = ds.shuffle(buffer_size=buffer_size)
    if batch_size is not None:
        drop_remainder = True if train_test != "test" else False
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=batch_size)
    return ds

class FlowerModel(tf.keras.Model):
    def __init__(self,
                 image_shape: Tuple[int, int],
                 classes: int,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        imagey, imagex, _ = image_shape
        #from tensorflow.keras.applications import DenseNet201
        from tensorflow.keras.applications import Xception
        #self.rnet = DenseNet201(
        #        input_shape=(imagey, imagex, 3),
        #        weights="imagenet",
        #        include_top=False,)
        self.rnet = Xception(
                input_shape=(imagey, imagex, 3),
                weights="imagenet",
                include_top=False,)
        self.rnet.trainable = False
        self.pooling = tf.keras.layers.GlobalAveragePooling2D(
                name="pooling")
        self.flat = tf.keras.layers.Flatten(
                name="flatten_pooling")
        self.dense_hidden = tf.keras.layers.Dense(
                units=4000,
                activation="relu",
                name="dense_hidden")
        self.dropout = tf.keras.layers.Dropout(
                0.25,
                name="dropout_layer")
        self.out_layer = tf.keras.layers.Dense(
                classes,
                activation="softmax",
                dtype=tf.float32,
                name="flower_class")
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        opt = tf.keras.optimizers.Adam(
                learning_rate=1e-3)
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
                learning_rate=1e-5)
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

def get_model(
              imagey: int,
              imagex: int,
              classes: int,
              seed: int=50,
              ) -> tf.keras.Model:
    tf.random.set_seed(seed)
    model = FlowerModel(
             image_shape=(imagey, imagex, 3),
             classes=classes,
            name="flower_model")
    return model

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
        epochs_init: int=3,
        epochs_tune: int=5,
        repeat: int=3,
        buffer_size: int=10):
    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    print(f"TF version: {tf.__version__}")
    classes = 104
    tpu, strategy = get_strategy()
    on_tpu = True if tpu else False
    print(f"TPU: {tpu}")
    print(f"Strategy: {strategy}")
    dims = [192, 224, 331, 512]
    dim = dims[1]
    print(f"Image dimension: {dim}")
    print(f"Epochs initial: {epochs_init}")
    print(f"Epochs tuning: {epochs_tune}")
    batch_size = batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        ds_train = get_dataset(dim,
                    repeat=repeat,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    apply_transformation=True,
                    train_test="train",)
        ds_valid = get_dataset(
                    dim,
                    repeat=None,
                    buffer_size=None,
                    batch_size=batch_size,
                    apply_transformation=False,
                    train_test="val",)
        model = get_model(
                    imagey=dim,
                    imagex=dim,
                    classes=classes,)
        print("Initial training: DenseNet not trainable")
        hist0 = model.fit(ds_train,
                    validation_data=ds_valid,
                    epochs=epochs_init)
        print("Fine-tuning training: DenseNet trainable")
        model.set_trainable_recompile()
        early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2)
        hist1 = model.fit(ds_train,
                    validation_data=ds_valid,
                    epochs=epochs_tune,
                    callbacks=[early_stopping])
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
    #test_image_transform()
    model, history, prediction = main(
            batch_size=64,
            epochs_init=2,
            epochs_tune=5,
            repeat=3,
            buffer_size=10)
