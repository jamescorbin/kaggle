import sys
import os
from typing import Dict, Any
import tensorflow as tf

TWO_FIVE_FIVE = tf.constant(
        255.0,
        shape=[],
        dtype=tf.float32,
        name="255")

@tf.function
def cast_image_float(image: tf.Tensor) -> tf.Tensor:
    image = tf.divide(
                tf.cast(image, tf.float32),
                TWO_FIVE_FIVE)
    return image

def read_tfrecord(example,
                  dim: int=192,
                  labeled: bool=True,
                  ):
    tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string)}
    if labeled:
        tfrec_format["class"] = tf.io.FixedLenFeature(
                [], tf.int64)
    parsed = tf.io.parse_single_example(
            example,
            tfrec_format)
    image = cast_image_float(
            tf.reshape(
                tf.image.decode_jpeg(
                    parsed["image"], channels=3),
                (dim, dim, 3)))
    out = {
            "image": image,
            "id": tf.io.decode_raw(parsed["id"],
                                 out_type=tf.uint8,
                                 fixed_length=9)}
    if labeled:
        out["class"] = tf.cast(parsed["class"], tf.int64)
    return out

def get_training_dataset(
        config: Dict[str, Any],
        top_dir: str=os.path.join(os.getcwd(), "data"),
        ) -> tf.data.Dataset:
    if not os.path.exists(top_dir):
        from kaggle_datasets import KaggleDatasets
        top_dir=KaggleDatasets().get_gcs_path(
                "tpu-getting-started")
    dirname = lambda x: f"tfrecords-jpeg-{x}x{x}"
    dim = config["image_shape"][0]
    search_string = [
        rf"{top_dir}/{dirname(dim)}/"
            r"train/*.tfrec",
        rf"{top_dir}/{dirname(dim)}/"
            r"val/*.tfrec"]
    tf_recs = tf.io.gfile.glob(search_string)
    read_f = lambda x: read_tfrecord(x,
            dim=dim,
            labeled=True,)
    x_ds = tf.data.TFRecordDataset(tf_recs).map(read_f)
    return x_ds

def get_prediction_dataset(
        config: Dict[str, Any],
        top_dir: str=os.path.join(os.getcwd(), "data"),
        ) -> tf.data.Dataset:
    if not os.path.exists(top_dir):
        from kaggle_datasets import KaggleDatasets
        top_dir=KaggleDatasets().get_gcs_path(
                "tpu-getting-started")
    dirname = lambda x: f"tfrecords-jpeg-{x}x{x}"
    tf_recs = tf.io.gfile.glob(
            f"{top_dir}/{dirname(dim)}/"
            f"{test}/*.tfrec")
    read_f = lambda x: read_tfrecord(x,
            dim=dim,
            labeled=False,)
    ds = tf.data.TFRecordDataset(tf_recs).map(read_f)
    return ds
