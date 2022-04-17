import sys
import os
import logging
from typing import Dict
import tensorflow as tf
import pandas as pd
import numpy as np
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import rawdata

logger = logging.getLogger(name=__name__)

def _byteslist(value):
    return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=value))
def _int64list(value):
    return tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
def _floatlist(value):
    return tf.train.Feature(
            float_list=tf.train.FloatList(value=value))

def serialize_example(ds: Dict[str, np.array]):
    feature = {
        "customer_id": _byteslist(ds["customer_id"]),
        "article_id": _byteslist(ds["article_id"]),
        "article_id_hist": _byteslist(ds["article_id_hist"]),
        "t_dat": _byteslist(ds["t_dat"]),
        "price": _floatlist(ds["price"]),
        "sales_channel_id": _byteslist(ds["sales_channel_id"]),
        }
    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(ds):
    tf_string = tf.py_function(
        serialize_example,
        ds,
        tf.string)
    return tf.reshape(tf_string, ())

def parse(example, ts_len: int=5):
    feature_description = {
        "customer_id": tf.io.FixedLenFeature([1], tf.string),
        "article_id":
                tf.io.FixedLenFeature([1], tf.string),
        "article_id_hist":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "t_dat": tf.io.FixedLenFeature([1], tf.string),
        "price": tf.io.FixedLenFeature([2 * ts_len], tf.float32),
        "sales_channel_id":
                tf.io.FixedLenFeature([ts_len], tf.string),
        }
    return tf.io.parse_single_example(
            example,
            feature_description)

def write_chunk(
            transactions_ds: pd.DataFrame,
            tfrec_fn: str,
            tfrec_dir: str="tfrec",
            ts_len: int=5,):
    out_fp = os.path.join(tfrec_dir, tfrec_fn)
    logger.info(f"Writing {out_fp}")
    with tf.io.TFRecordWriter(out_fp) as writer:
        for i, row in transactions_ds.iterrows():
            data = {}
            data["article_id_hist"] = [
                    row[f"article_id_{n}"]
                    for n in range(ts_len, 0, -1)]
            data["article_id"] = [row["article_id"]]
            data["t_dat"] = [row["t_dat"]]
            data["price"] = (
                    [row[f"price_{n}"]
                    for n in range(ts_len, 0, -1)]
                    + [row[f"price_{n}_mask"]
                    for n in range(ts_len, 0, -1)])
            data["sales_channel_id"] = [
                    row[f"sales_channel_id_{n}"]
                    for n in range(ts_len, 0, -1)]
            data["customer_id"] = [row["customer_id"]]
            writer.write(serialize_example(data))

def write_dataset(
            transactions_fn: str,
            tfrec_dir: str="tfrec",
            ts_len: int=5,
            filesize: int=1_000_000):
    if not os.path.exists(tfrec_dir):
        os.mkdir(tfrec_dir)
    shards = 32
    for i in range(shards):
        chunk = rawdata.load_transactions_ds(
                transactions_fn=transactions_fn,
                skiprows=i * filesize)
        write_chunk(
                chunk,
                tfrec_dir=tfrec_dir,
                ts_len=ts_len,
                tfrec_fn=f"{i:03d}.tfrec")
