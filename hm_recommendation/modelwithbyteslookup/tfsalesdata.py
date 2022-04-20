import sys
import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import rawdata
import serialize

unk = rawdata.unk

def _make_hash_tables(
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame,
        ) -> pd.DataFrame:
    article_id = tf.constant(articles_ds.index.values)
    article_columns = [
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
            "detail_desc"]
    lookups = {
        col: tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                    article_id,
                    tf.constant(articles_ds[col].values)),
            default_value=unk.encode("utf-8"))
        for col in article_columns}
    lookups.update({
        col: tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                    tf.constant(customers_ds.index.values),
                    tf.constant(customers_ds[col].values)),
            default_value=unk.encode("utf-8"))
        for col in ["club_member_status", "fashion_news_frequency"]})
    lookups["age"] = (
            tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                        tf.constant(customers_ds.index.values),
                        tf.constant(customers_ds["age"].values)),
                default_value=0.0))
    lookups["article_id"] = (
            tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    article_id,
                    tf.range(0, len(articles_ds), 1)),
                default_value=0))
    lookups["customer_id"] = (
            tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    customers_ds.index.values,
                    tf.range(0, len(customers_ds), 1)),
                default_value=0))
    return lookups

def make_articles_tf(
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame):
    lookups = _make_hash_tables(articles_ds, customers_ds)
    articles_tf = tf.data.Dataset.from_tensor_slices(
        {
            "article_id": articles_ds.index.values,
            "product_type_name":
                articles_ds["product_type_name"].values,
            "product_group_name":
                articles_ds["product_group_name"].values,
            "graphical_appearance_name":
                articles_ds["graphical_appearance_name"].values,
            "colour_group_name":
                articles_ds["colour_group_name"].values,
            "perceived_colour_value_name":
                articles_ds["perceived_colour_value_name"].values,
            "perceived_colour_master_name":
                articles_ds["perceived_colour_master_name"].values,
            "department_name":
                articles_ds["department_name"].values,
            "index_name":
                articles_ds["index_name"].values,
            "index_group_name":
                articles_ds["index_group_name"].values,
            "section_name":
                articles_ds["section_name"].values,
            "garment_group_name":
                articles_ds["garment_group_name"].values,
            "detail_desc":
                articles_ds["detail_desc"].values,})
    return lookups, articles_tf

def make_tfds(
        tfrec_dir: str,
        lookups: Dict[str, Any],
        articles_tf: tf.data.Dataset,
        config: Dict[str, Any],
        ts_len: int=5):
    tfrec_files = [os.path.join(tfrec_dir, f)
                   for f in os.listdir(tfrec_dir)]
    filenames = tf.data.Dataset.from_tensor_slices(tfrec_files)
    transactions_tf = (
        filenames
            .interleave(
                lambda filename: tf.data.TFRecordDataset(filename)
                    .map(serialize.parse, num_parallel_calls=1),
                cycle_length=4,
                num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config["batch_size"], drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            )
    return transactions_tf, articles_tf
