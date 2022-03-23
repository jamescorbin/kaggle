"""
kaggle competitions download -c tpu-getting-started
"""
import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from typing import Optional, Tuple
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
    label = tf.cast(parsed["class"], tf.int64)
    idx = parsed["id"]
    return {"image": image, "id": idx, "label": label}

@tf.function
def trans(x):
    return {"image": transform(x["image"]),
            "id": x["id"],
            "label": x["label"]}

def get_strategy() -> Tuple[Optional["tpu"], Optional["strategy"]]:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    return tpu, strategy

def main():
    repeat = 3
    buffer_size = 1000
    classes = 103
    tpu, strategy = get_strategy()
    print(f"TPU: {tpu}")
    print(f"Strategy: {strategy}")
    dims = [192, 224, 331, 512]
    dirs = [f"tfrecords-jpeg-{dim}x{dim}" for dim in dims]
    parent_dir = os.path.join(os.getcwd(), "data")
    data_dirs = [os.path.join(parent_dir, tdir) for tdir in dirs]
    ds_fns = [os.path.join(data_dirs[0], "train", t)
              for t in os.listdir(os.path.join(data_dirs[0], "train"))]
    ds = tf.data.TFRecordDataset(ds_fns).map(read_tfrecord)
    ds = ds.repeat(repeat).shuffle(buffer_size=buffer_size).map(trans)
    print(ds)
    model = get_model(strategy, dims[0], dims[0], classes)
    #x = next(iter(ds))
    #print(transform(x["image"]))





from tensorflow.keras.applications import DenseNet201

def get_model(strategy: "strategy",
              imagex: int,
              imagey: int,
              classes: int) -> tf.keras.Model:
    with strategy.scope():
        rnet = DenseNet201(
            input_shape=(imagex, imagey, 3),
            weights='imagenet',
            include_top=False)
        # trainable rnet
        rnet.trainable = True
        input1 = tf.keras.Input(shape=(imagex, imagey, 3), dtype=tf.float32)
        input2 = tf.keras.Input(shape=(), dtype=tf.string)
        inputs = {"image": input1,
                  "id": input2}
        #input1 = tf.keras.Input(shape=(imagex, imagey, 3))
        pooling = tf.keras.layers.GlobalAveragePooling2D()
        out_layer = tf.keras.layers.Dense(classes,
                                          activation='softmax',
                                          dtype='float32')
        x = rnet(input1)
        x = pooling(x)
        x = out_layer(x)
        outputs = {"label": x}
        model = tf.keras.Model(inputs=inputs,
                               outputs=outputs,
                               name="aaa")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"])
    return model


if __name__=="__main__":
    main()
