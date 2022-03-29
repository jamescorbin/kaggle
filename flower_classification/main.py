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
import tensorflow_datasets as tfds
from kaggle_datasets import KaggleDatasets

@tf.function
def transform(x: tf.Tensor) -> tf.Tensor:
    return apply_affine_transform(x, get_random_transformation())

@tf.function
def trans(x: tf.Tensor) -> tf.Tensor:
    return {"image": transform(x["image"]), "id": x["id"]}

@tf.function
def get_random_rotation(th0: tf.Tensor=tf.constant(15.0)) -> tf.Tensor:
    rot = tf.random.normal([1],
                        mean=0,
                        stddev= th0 * 2.0 * np.pi / 360.0,
                        dtype=tf.float32)
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    sn = tf.math.sin(rot)
    cn = tf.math.cos(rot)
    rot_matrix = tf.reshape(
                    [cn, sn, zero,
                    -1.0 * sn, cn, zero,
                    zero, zero, one],
                    (3, 3))
    return rot_matrix

@tf.function
def get_random_shear(shr0: tf.Tensor=tf.constant(15.0)) -> tf.Tensor:
    shr = tf.random.normal([1],
                        mean=0,
                        stddev= shr0 * 2.0 * np.pi / 360.0,
                        dtype=tf.float32)
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    sn = tf.math.sin(shr)
    cn = tf.math.cos(shr)
    shear_matrix = tf.reshape(
                [cn * sn + 1.0, cn, zero,
                sn, one, zero,
                zero, zero, one],
                (3, 3))
    return shear_matrix

@tf.function
def get_random_shift(
                xd0: tf.Tensor=tf.constant(10.0),
                yd0: tf.Tensor=tf.constant(10.0),
                ) -> tf.Tensor:
    h_shift = tf.random.normal([1],
                                     mean=1.0,
                                     stddev=xd0,
                                     dtype=tf.float32)
    w_shift = tf.random.normal([1],
                                     mean=1.0,
                                     stddev=yd0,
                                     dtype=tf.float32)
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    shift_matrix = tf.reshape(
                [one, zero, h_shift,
                zero, one, w_shift,
                zero, zero, one],
                (3, 3))
    return shift_matrix

@tf.function
def get_random_scale(
                sclx0: tf.Tensor=tf.constant(10.0),
                scly0: tf.Tensor=tf.constant(10.0),
                ) -> tf.Tensor:
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    h_zoom = tf.random.normal([1], stddev=sclx0, dtype=tf.float32)
    w_zoom = tf.random.normal([1], stddev=scly0, dtype=tf.float32)
    zoom_matrix = tf.reshape(
                [1.0 / h_zoom, zero, zero,
                zero, 1.0 / w_zoom, zero,
                zero, zero, one],
                (3, 3))
    return zoom_matrix

@tf.function
def get_random_transformation(
                th0: tf.Tensor=tf.constant(15.0),
                shr0: tf.Tensor=tf.constant(15.0),
                xd0: tf.Tensor=tf.constant(10.0),
                yd0: tf.Tensor=tf.constant(10.0),
                sclx0: tf.Tensor=tf.constant(10.0),
                scly0: tf.Tensor=tf.constant(10.0),
                ) -> tf.Tensor:
    rot_matrix = get_random_rotation(th0)
    shear_matrix = get_random_shear(shr0)
    shift_matrix = get_random_shift(xd0, yd0)
    zoom_matrix = get_random_scale(sclx0, scly0)
    return tf.linalg.matmul(rot_matrix,
            tf.linalg.matmul(shear_matrix,
                tf.linalg.matmul(zoom_matrix,
                    shift_matrix)))

@tf.function
def apply_affine_transform(img: tf.Tensor,
                           mat: tf.Tensor,
                           ) -> tf.Tensor:
    """
    This method was used due to the
    tf.keras.preprocessing.image.apply_affine_transformation
    being a numpy function making that not amenable
    to launching on a TPU.
    There are methods like tf.keras.layers.RandomRotation
    that solve this issue.
    """
    zero = tf.constant(0, dtype=tf.int32)
    dim = tf.gather(tf.shape(img), zero)
    d = tf.cast(dim // 2, tf.float32)
    iidxs = (tf.tile(tf.reshape(tf.range(dim, dtype=tf.float32),
                           (1, dim)), [dim, 1]) - d)
    jidxs = (tf.tile(tf.reshape(tf.range(dim, dtype=tf.float32),
                           (dim, 1)), [1, dim]) - d)
    affidx = tf.ones(shape=(dim, dim))
    idxs = tf.stack([iidxs, jidxs, affidx], axis=2)
    k1 = tf.tensordot(idxs, mat, axes=[[-1], [-1]])
    k2 = tf.cast(k1 + d, dtype=tf.int32)
    k3 = tf.reshape(k2, (dim * dim, 3))
    k4 = tf.clip_by_value(k3, 0, dim - 1)
    k5 = tf.slice(k4, (0, 0), (dim * dim, 2))
    gat = tf.gather_nd(params=img, indices=k5)
    new_img = tf.reshape(gat, (dim, dim, 3))
    return new_img

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
                  on_tpu: bool=False):
    tfrec_format = {"image": tf.io.FixedLenFeature([], tf.string)}
    if not on_tpu:
        tfrec_format["id"] = tf.io.FixedLenFeature([], tf.string)
    if labeled:
        tfrec_format["class"] = tf.io.FixedLenFeature([], tf.int64)
    parsed = tf.io.parse_single_example(example, tfrec_format)
    image = tf.reshape(
                tf.cast(
                    tf.image.decode_jpeg(parsed["image"], channels=3),
                    tf.float32),
                (dim, dim, 3))
    out = {"image": image}
    if not on_tpu:
        out["id"] = parsed["id"]
    if labeled:
        out["class"] = tf.cast(parsed["class"], tf.int64)
    return out

def get_dataset(dim: int,
                repeat: Optional[int],
                buffer_size: Optional[int],
                batch_size: Optional[int],
                train_test: str="train",
                apply_transformation: bool=False,
                #top_dir: str=os.path.join(os.getcwd(), "data"),
                top_dir=KaggleDatasets().get_gcs_path("tpu-getting-started"),
                on_tpu: bool=False,
                ) -> tf.data.Dataset:
    dirname = lambda x: f"tfrecords-jpeg-{x}x{x}"
    tf_recs = tf.io.gfile.glob(f"{top_dir}/"
                               f"{dirname(dim)}/"
                               f"{train_test}/*.tfrec")
    if train_test in ("train", "val"):
        read_f = lambda x: read_tfrecord(x,
                                         dim=dim,
                                         labeled=True,
                                         on_tpu=on_tpu)
    else:
        read_f = lambda x: read_tfrecord(x,
                                         dim=dim,
                                         labeled=False,
                                         on_tpu=on_tpu)
    ds = tf.data.TFRecordDataset(tf_recs).map(read_f)
    if repeat is not None:
        ds = ds.repeat(repeat)
    if buffer_size is not None:
        ds = ds.shuffle(buffer_size=buffer_size)
    if train_test in ("train", "val"):
        if on_tpu:
            xds = ds.map(lambda x: {"image": x["image"]})
            if apply_transformation:
                xds = ds.map(lambda x: {"image": transform(x["image"])})
        else:
            xds = ds.map(lambda x: {"image": x["image"], "id": x["id"]})
            if apply_transformation:
                xds = xds.map(trans)
        yds = ds.map(lambda x: {"class": x["class"]})
        ds = tf.data.Dataset.zip((xds, yds))
    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds

def get_model(strategy: "strategy",
              imagey: int,
              imagex: int,
              classes: int,
              seed: int=50,
              on_tpu: bool=False) -> tf.keras.Model:
    with strategy.scope():
        tf.random.set_seed(seed)
        """
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
        """
        rnet = DenseNet201(
                input_shape=(imagey, imagex, 3),
                weights="imagenet",
                include_top=False,)
        input1 = tf.keras.Input(
                shape=(imagey, imagex, 3),
                dtype=tf.float32)
        inputs = {"image": input1}
        if not on_tpu:
            input2 = tf.keras.Input(shape=(), dtype=tf.string)
            inputs["id"] = input2
        pooling = tf.keras.layers.GlobalAveragePooling2D(
                name="pooling")
        out_layer = tf.keras.layers.Dense(
                classes,
                activation="softmax",
                dtype=tf.float32)
        x = input1
        """
        x = random_translation_layer(x)
        x = random_flip_layer(x)
        x = random_rotation_layer(x)
        #x = random_shear_height(x)
        #x = center_crop_layer(x)
        #x = random_shear_width(x)
        #x = center_crop_layer(x)
        x = random_zoom_layer(x)
        x = center_crop_layer(x)
        """
        x = rnet(x)
        x = pooling(x)
        x = out_layer(x)
        label = tf.reshape(tf.math.top_k(x, k=1).indices, shape=[-1])
        outputs = {"class": x, "label": label}
        if not on_tpu:
            outputs["id"] = input2
        model = tf.keras.Model(
                inputs=inputs,
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

def main(
        batch_size: int=256,
        epochs_init: int=3,
        epochs_tune: int=5,
        repeat: int=3,
        buffer_size: int=10):
    print(f"TF version: {tf.__version__}")
    classes = 104
    tpu, strategy = get_strategy()
    on_tpu = True if tpu else False
    print(f"TPU: {tpu}")
    print(f"Strategy: {strategy}")
    dims = [192, 224, 331, 512]
    dim = dims[0]
    print(f"Image dimension: {dim}")
    print(f"Epochs initial: {epochs_init}")
    print(f"Epochs tuning: {epochs_tune}")
    ds_train = get_dataset(dim,
                    repeat=repeat,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    apply_transformation=True,
                    train_test="train",
                    on_tpu=on_tpu)
    ds_valid = get_dataset(
                    dim,
                    repeat=None,
                    buffer_size=None,
                    batch_size=batch_size,
                    apply_transformation=False,
                    train_test="val",
                    on_tpu=on_tpu)
    model = get_model(strategy=strategy,
                    imagey=dim,
                    imagex=dim,
                    classes=classes,
                    on_tpu=on_tpu)
    print("Initial training: DenseNet not trainable")
    model.get_layer("densenet201").trainable = False
    hist0 = model.fit(ds_train,
                    validation_data=ds_valid,
                    epochs=epochs_init)
    print("Fine-tuning training: DenseNet trainable")
    model.get_layer("densenet201").trainable = True
    hist1 = model.fit(ds_train,
                    validation_data=ds_valid,
                    epochs=epochs_tune)
    ds_test = get_dataset(
                    dim,
                    repeat=None,
                    buffer_size=None,
                    batch_size=None,
                    apply_transformation=False,
                    train_test="test",
                    on_tpu=False)
    if on_tpu:
        ids = tfds.as_numpy(ds_test.map(lambda x: {"id": x["id"]}))
        ids = np.array([x["id"] for x in ids])
        ds_test = ds_test.map(lambda x: {"image": x["image"]})
        prediction = model.predict(ds_test.batch(64))
        prediction.pop("class")
        prediction = pd.DataFrame(prediction)
        prediction["id"] = ids
    else:
        prediction = model.predict(ds_test.batch(64))
        prediction.pop("class")
        prediction = pd.DataFrame(prediction)
    return model, (hist0, hist1), prediction

if __name__=="__main__":
    model, history, prediction = ds_test = main(
            batch_size=256,
            epochs_init=3,
            epochs_tune=5,
            repeat=3,
            buffer_size=10)
