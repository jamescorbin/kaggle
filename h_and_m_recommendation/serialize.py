import sys
import os
import json
import logging
from typing import Dict, List, Any
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
        #"t_dat": _byteslist(ds["t_dat"]),
        "test": _int64list(ds["test"]),
        "article_id": _int64list(ds["article_id"]),
        "article_id_hist": _int64list(ds["article_id_hist"]),
        "sales_channel_id": _int64list(ds["sales_channel_id"]),
        "price": _floatlist(ds["price"]),
        "price_mask": _floatlist(ds["price_mask"]),
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

def parse(example, ts_len: int):
    feature_description = {
        "customer_id": tf.io.FixedLenFeature([1], tf.string),
        #"t_dat": tf.io.FixedLenFeature([1], tf.string),
        "test": tf.io.FixedLenFeature([1], tf.int64),
        "article_id":
                tf.io.FixedLenFeature([1], tf.int64),
        "article_id_hist":
                tf.io.FixedLenFeature([ts_len], tf.int64),
        "sales_channel_id":
                tf.io.FixedLenFeature([ts_len], tf.int64),
        "price": tf.io.FixedLenFeature([ts_len], tf.float32),
        "price_mask": tf.io.FixedLenFeature([ts_len], tf.float32),
        }
    return tf.io.parse_single_example(
            example,
            feature_description)

def write_chunk(
            transactions_ds: pd.DataFrame,
            index: int,
            filesize: int,
            tfrec_fn: str,
            tfrec_dir: str,
            ts_len: int,):
    out_fp = os.path.join(tfrec_dir, tfrec_fn)
    logger.info(f"Writing {out_fp}")
    idx_range = range(index * filesize,
                      min((index + 1) * filesize, len(transactions_ds)))
    with tf.io.TFRecordWriter(out_fp) as writer:
        for i, row in transactions_ds.iloc[idx_range].iterrows():
            data = rawdata.convert_transaction_to_datapoint(row, ts_len)
            writer.write(serialize_example(data))

def write_dataset(
            transactions_fn: str,
            tfrec_dir: str="tfrec",
            ts_len: int=4,
            filesize: int=1_000_000):
    if not os.path.exists(tfrec_dir):
        os.mkdir(tfrec_dir)
    shards = 34
    transactions_ds = pd.read_parquet(transactions_fn)
    for i in range(shards):
        write_chunk(
                transactions_ds,
                index=i,
                filesize=filesize,
                tfrec_dir=tfrec_dir,
                ts_len=ts_len,
                tfrec_fn=f"{i:03d}.tfrec")

if __name__=="__main__":
    config_fn = "config-model.json"
    tfrec_dir = "data/tfrec"
    with open(config_fn, "r") as f:
        config = json.load(f)
    sequential_fn = "data/sequential.parquet"
    write_dataset(
            sequential_fn,
            tfrec_dir=tfrec_dir,
            ts_len=config["ts_len"],)
