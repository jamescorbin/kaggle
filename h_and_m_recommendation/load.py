import sys
import os
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import rawdata
import serialize

def make_articles_tf(articles_ds: pd.DataFrame) -> tf.data.Dataset:
    articles_tf = tf.data.Dataset.from_tensor_slices(
        {
            "article_id": articles_ds["article_id"].values,
            "product_group_name":
                articles_ds["product_group_name"].values,
            "graphical_appearance_name":
                articles_ds["graphical_appearance_name"].values,
            "perceived_colour_master_name":
                articles_ds["perceived_colour_master_name"].values,
            "section_name":
                articles_ds["section_name"].values,
            #"product_type_name":
            #    articles_ds["product_type_name"].values,
            #"colour_group_name":
            #    articles_ds["colour_group_name"].values,
            #"perceived_colour_value_name":
            #    articles_ds["perceived_colour_value_name"].values,
            #"department_name":
            #    articles_ds["department_name"].values,
            #"index_name":
            #    articles_ds["index_name"].values,
            #"index_group_name":
            #    articles_ds["index_group_name"].values,
            #"garment_group_name":
            #    articles_ds["garment_group_name"].values,
            #"detail_desc":
            #    articles_ds["detail_desc"].values,
        })
    return articles_tf

def make_tfds(
        tfrec_dir: str,
        config: Dict[str, Any],
        ts_len: int,
        ):
    _f = lambda x, y: y
    _train = lambda x: x["test"][0] == tf.constant(0, dtype=tf.int64)
    _g1 = lambda x, y: x % config["mod_k"] <= config["n_train"]
    _g2 = (lambda x, y:
           (config["n_train"] < x % config["mod_k"])
           & (x % config["mod_k"] <= config["n_valid"]))
    _g3 = lambda x, y: (x % config["mod_k"]) > config["n_valid"]
    parse_f = lambda x: serialize.parse(x, ts_len=config["ts_len"])
    tfrec_files = [os.path.join(tfrec_dir, f)
                   for f in os.listdir(tfrec_dir)]
    filenames = tf.data.Dataset.from_tensor_slices(tfrec_files)
    transactions_tf = (
        filenames
            .interleave(
                lambda filename: tf.data.TFRecordDataset(filename)
                    .map(parse_f, num_parallel_calls=2),
                cycle_length=4,
                num_parallel_calls=tf.data.AUTOTUNE)
            .filter(_train))
    xtrain = (
        transactions_tf
            .enumerate()
            .filter(_g1)
            .map(_f))
    xvalid = (
        transactions_tf
            .enumerate()
            .filter(_g2)
            .map(_f))
    xtest = (
        transactions_tf
            .enumerate()
            .filter(_g3)
            .map(_f))
    return xtrain, xvalid, xtest

def get_prediction_data(
        tfrec_dir: str,
        config: Dict[str, Any],
        ):
    _train = lambda x: x["test"][0] == tf.constant(1, dtype=tf.int64)
    parse_f = lambda x: serialize.parse(x, ts_len=config["ts_len"])
    tfrec_files = [os.path.join(tfrec_dir, f)
                   for f in os.listdir(tfrec_dir)]
    filenames = tf.data.Dataset.from_tensor_slices(tfrec_files)
    transactions_tf = (
        filenames
            .interleave(
                lambda filename: tf.data.TFRecordDataset(filename)
                    .map(parse_f, num_parallel_calls=2),
                cycle_length=4,
                num_parallel_calls=tf.data.AUTOTUNE)
            .filter(_train))
    return transactions_tf


