"""
kaggle competitions download -c tpu-getting-started
"""
import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from tensorflow.keras.applications import DenseNet201
pt = os.getcwd()
sys.path.insert(1, pt)
from augmentimgs import transform

def read_tfrecord(example, dim: int=192, labeled: bool=True):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),}
    if labeled:
        tfrec_format["class"] = tf.io.FixedLenFeature([], tf.int64)
    parsed = tf.io.parse_single_example(example, tfrec_format)
    image = tf.reshape(
                tf.cast(
                    tf.image.decode_jpeg(parsed["image"], channels=3),
                    tf.float32),
                (dim, dim, 3))
    idx = parsed["id"]
    out = {"image": image, "id": idx}
    if labeled:
        label = tf.cast(parsed["class"], tf.int64)
        out["class"] = label
    return out

@tf.function
def trans(x):
    return {"image": transform(x["image"]),
            "id": x["id"],
            "class": x["class"]}

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

def get_model(strategy: "strategy",
              imagex: int,
              imagey: int,
              classes: int,
              seed: int=50) -> tf.keras.Model:
    with strategy.scope():
        tf.random.set_seed(seed)
        one = tf.constant([1], dtype=tf.int32)
        dimx = tf.constant(imagex, dtype=tf.int32)
        dimy = tf.constant(imagey, dtype=tf.int32)
        dx_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        dy_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        rot_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        shear_y_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.10,
                    dtype=tf.float32))
        shear_x_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.10,
                    dtype=tf.float32))
        zoom_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        random_translation_layer = tf.keras.layers.RandomTranslation(
                    dx_0,
                    dy_0,
                    name="random_translation")
        random_flip_layer = tf.keras.layers.RandomFlip(
                    name="random_flip")
        random_rotation_layer = tf.keras.layers.RandomRotation(
                    rot_0,
                    name="random_rotation")
        random_shear_height = tf.keras.layers.RandomHeight(
                    factor=shear_y_0,
                    name="random_shear_height")
        random_shear_width = tf.keras.layers.RandomWidth(
                    factor=shear_x_0,
                    name="random_shear_width")
        random_zoom_layer = tf.keras.layers.RandomZoom(
                    height_factor=zoom_0,
                    width_factor=zoom_0,
                    name="random_zoom")
        center_crop_layer = tf.keras.layers.CenterCrop(
                    #height=dimy,
                    #width=dimx,
                    height=imagey,
                    width=imagex,
                    name="center_crop")
        rnet = DenseNet201(
            input_shape=(imagey, imagex, 3),
            weights="imagenet",
            include_top=False,)
        input1 = tf.keras.Input(shape=(imagey, imagex, 3), dtype=tf.float32)
        input2 = tf.keras.Input(shape=(), dtype=tf.string)
        inputs = {"image": input1,
                  "id": input2}
        pooling = tf.keras.layers.GlobalAveragePooling2D(
                    name="pooling")
        out_layer = tf.keras.layers.Dense(
                    classes,
                    activation="softmax",
                    dtype=tf.float32)
        x = input1
        x = random_translation_layer(x)
        x = random_flip_layer(x)
        x = random_rotation_layer(x)
        #x = random_shear_height(x)
        #x = center_crop_layer(x)
        #x = random_shear_width(x)
        #x = center_crop_layer(x)
        x = random_zoom_layer(x)
        x = center_crop_layer(x)
        x = rnet(x)
        x = pooling(x)
        x = out_layer(x)
        outputs = {"class": x, "id": input2}
        model = tf.keras.Model(inputs=inputs,
                               outputs=outputs,
                               name="flower_model")
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    opt = tf.keras.optimizers.Adam()
    model.compile(
            optimizer=opt,
            loss={"class": loss},
            metrics={"class": [metric]})
    return model

def get_dataset(dim: int,
                repeat: Optional[int],
                buffer_size: Optional[int],
                batch_size: Optional[int],
                train_test: str="train"
                ) -> tf.data.Dataset:
    dirname = lambda x: f"tfrecords-jpeg-{x}x{x}"
    parent_dir = os.path.join(os.getcwd(), "data", dirname(dim), train_test)
    ds_fns = [os.path.join(parent_dir, t) for t in os.listdir(parent_dir)]
    read_f = lambda x: read_tfrecord(x, dim=dim)
    ds = tf.data.TFRecordDataset(ds_fns).map(read_f)
    if repeat is not None:
        ds = ds.repeat(repeat)
    if buffer_size is not None:
        ds = ds.shuffle(buffer_size=buffer_size)
    if train_test == "train":
        xds = ds.map(lambda x: {"image": x["image"], "id": x["id"]})
        yds = ds.map(lambda x: {"class": x["class"]})
        ds = tf.data.Dataset.zip((xds, yds))
    if batch_size is not None:
        ds = ds.batch(batch_size)
    return ds

def main():
    repeat = 1
    buffer_size = 100
    classes = 104
    batch_size = 32
    tpu, strategy = get_strategy()
    print(f"TPU: {tpu}")
    print(f"Strategy: {strategy}")
    dims = [192, 224, 331, 512]
    dim = dims[0]
    ds = get_dataset(dim,
                     repeat=repeat,
                     buffer_size=buffer_size,
                     batch_size=batch_size)
    print(ds)
    model = get_model(strategy, dim, dim, classes)
    print("Initial training: DenseNet not trainable")
    model.get_layer("densenet201").trainable = False
    hist0 = model.fit(ds, epochs=2)
    print("Fine-tuning training: DenseNet trainable")
    model.get_layer("densenet201").trainable = True
    hist1 = model.fit(ds, epochs=2)
    ds_test = get_dataset(
                    dim,
                    repeat=None,
                    buffer_size=None,
                    batch_size=None,
                    test_train="test")
    prediction = model.predict(ds_test)
    print(prediction)
    prediction = pd.DataFrame(prediction)
    prediction.rename({"class": "label"}, axis=1, inplace=True)
    print(prediction)

if __name__=="__main__":
    main()
