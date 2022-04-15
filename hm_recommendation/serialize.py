import sys
import os
from typing import Dict
import tensorflow as tf
import pandas as pd
import numpy as np

def _byteslist(value):
    return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=value))
def _int64list(value): return tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
def _floatlist(value):
    return tf.train.Feature(
            float_list=tf.train.FloatList(value=value))

def serialize_example(ds: Dict[str, np.array]):
    feature = {
        "t_dat": _byteslist(ds["t_dat"]),
        "customer_id": _byteslist(ds["customer_id"]),
        "fn": _floatlist(ds["FN"]),
        "active": _floatlist(ds["Active"]),
        "age": _floatlist(ds["age"]),
        "price": _floatlist(ds["price"]),
        "club_member_status":
                _byteslist(ds["club_member_status"]),
        "fashion_news_frequency":
                _byteslist(ds["fashion_news_frequency"]),
        "sales_channel_id": _byteslist(ds["sales_channel_id"]),
        "article_id": _byteslist(ds["article_id"]),
        "product_type_name":
                _byteslist(ds["product_type_name"]),
        "product_group_name":
                _byteslist(ds["product_group_name"]),
        "graphical_appearance_name":
                _byteslist(ds["graphical_appearance_name"]),
        "colour_group_name":
                _byteslist(ds["colour_group_name"]),
        "perceived_colour_value_name":
                _byteslist(ds["perceived_colour_value_name"]),
        "perceived_colour_master_name":
                _byteslist(ds["perceived_colour_master_name"]),
        "department_name":
                _byteslist(ds["department_name"]),
        "index_name":
                _byteslist(ds["index_name"]),
        "index_group_name":
                _byteslist(ds["index_group_name"]),
        "section_name":
                _byteslist(ds["section_name"]),
        "garment_group_name":
                _byteslist(ds["garment_group_name"]),
        #"detail_desc":
        #        _byteslist(ds["detail_desc"]),
        "target_article_id":
                _byteslist(ds["target_article_id"]),
        "target": _int64list(ds["target"]),}
    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(ds):
    tf_string = tf.py_function(
        serialize_example,
        ds,
        tf.string)
    return tf.reshape(tf_string, ())

def parse(example):
    ts_len = 5
    feature_description = {
        "t_dat": tf.io.FixedLenFeature([1], tf.string),
        "customer_id": tf.io.FixedLenFeature([1], tf.string),
        "fn": tf.io.FixedLenFeature([2], tf.float32),
        "active": tf.io.FixedLenFeature([2], tf.float32),
        "age": tf.io.FixedLenFeature([2], tf.float32),
        "price": tf.io.FixedLenFeature([2 * ts_len], tf.float32),
        "club_member_status":
                tf.io.FixedLenFeature([1], tf.string),
        "fashion_news_frequency":
                tf.io.FixedLenFeature([1], tf.string),
        "sales_channel_id":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "article_id":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "product_type_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "product_group_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "graphical_appearance_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "colour_group_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "perceived_colour_value_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "perceived_colour_master_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "department_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "index_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "index_group_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "section_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        "garment_group_name":
                tf.io.FixedLenFeature([ts_len], tf.string),
        #"detail_desc": tf.io.FixedLenFeature([ts_len], tf.string),
        "target_article_id":
                tf.io.FixedLenFeature([1], tf.string),
        "target": tf.io.FixedLenFeature([1], tf.int64),}
    return tf.io.parse_single_example(example, feature_description)

def write_chunk(
            transactions_ds: pd.DataFrame,
            articles_ds: pd.DataFrame,
            tfrec_fn: str,
            tfrec_dir: str="tfrec",
            ts_len: int=5):
    dataset = []
    article_columns = [
            "product_code",
            "prod_name",
            "product_type_no",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_no",
            "graphical_appearance_name",
            "colour_group_code",
            "colour_group_name",
            "perceived_colour_value_id",
            "perceived_colour_value_name",
            "perceived_colour_master_id",
            "perceived_colour_master_name",
            "department_no",
            "department_name",
            "index_code",
            "index_name",
            "index_group_no",
            "index_group_name",
            "section_no",
            "section_name",
            "garment_group_no",
            "garment_group_name",
            "detail_desc"]
    for i in range(1, ts_len + 1):
        transactions_ds = transactions_ds.join(
                articles_ds,
                on=f"article_id_{i}",
                how="left",
                rsuffix=f"_{i}")
        transactions_ds.rename(
                {col: f"{col}_{i}"
                 for col in article_columns},
                axis=1,
                inplace=True)
    out_fp = os.path.join(tfrec_dir, tfrec_fn)
    with tf.io.TFRecordWriter(out_fp) as writer:
        for i, row in transactions_ds.iterrows():
            data = {}
            data["article_id"] = [
                    row[f"article_id_{n}"]
                    for n in range(ts_len, 0, -1)]
            for col in article_columns:
                data[col] = [row[f"{col}_{n}"] for n in range(ts_len, 0, -1)]
            data["t_dat"] = [row["t_dat"]]
            data["price"] = (
                    [row[f"price_{n}"]
                    for n in range(ts_len, 0, -1)]
                    + [row[f"price_{n}_mask"]
                    for n in range(ts_len, 0, -1)])
            data["sales_channel_id"] = [row[f"sales_channel_id_{n}"]
                    for n in range(ts_len, 0, -1)]
            data["target"] = [row["target"]]
            data["target_article_id"] = [row["article_id"]]
            cust_idx = row["customer_id"]
            data["customer_id"] = [cust_idx]
            data["FN"] = [row["FN"], row["fn_mask"]]
            data["Active"] = [row["Active"], row["active_mask"]]
            data["club_member_status"] = [row["club_member_status"]]
            data["fashion_news_frequency"] = [row["fashion_news_frequency"]]
            data["age"] = [row["age"], row["age_mask"]]
            writer.write(serialize_example(data))

def write_dataset(
            transactions_ds: pd.DataFrame,
            articles_ds: pd.DataFrame,
            customers_ds: pd.DataFrame,
            tfrec_dir: str="tfrec",
            ts_len: int=5,
            filesize: int=1000000):
    if not os.path.exists(tfrec_dir):
        os.mkdir(tfrec_dir)
    n = len(transactions_ds)
    shards = n // filesize + 1 if n % filesize != 0 else n // filesize
    transactions_ds = transactions_ds.join(
            customers_ds,
            on="customer_id",
            how="left")
    for i in range(shards):
        idx = range(i * filesize, (i + 1) * filesize)
        chunk = pd.DataFrame(transactions_ds.iloc[idx])
        write_chunk(
                chunk,
                articles_ds,
                tfrec_dir=tfrec_dir,
                ts_len=ts_len,
                tfrec_fn=f"{i:03d}.tfrec")




