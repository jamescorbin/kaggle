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

def _make_hash_tables(
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame,
        ) -> pd.DataFrame:
    article_id = tf.constant(articles_ds["article_id"].values)
    customer_id = tf.constant(customers_ds["customer_id"].values)
    article_columns = [
            "product_group_name",
            "graphical_appearance_name",
            "perceived_colour_master_name",
            "section_name",
            #"product_type_name",
            #"colour_group_name",
            #"perceived_colour_value_name",
            #"department_name",
            #"index_name",
            #"index_group_name",
            #"garment_group_name",
            #"detail_desc",
            ]
    lookups = {
        col: tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                    article_id,
                    tf.constant(articles_ds[col].values)),
            default_value=0)
        for col in article_columns}
    lookups.update({
        col: tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                    customer_id,
                    tf.constant(customers_ds[col].values)),
            default_value=0)
        for col in ["club_member_status", "fashion_news_frequency"]})
    lookups["age"] = (
            tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                        customer_id,
                        tf.constant(customers_ds["age"].values)),
                default_value=0.0))
    return lookups

def make_articles_tf(
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame,
        ) -> Tuple[Dict[str, "StaticHashTable"], tf.data.Dataset]:
    lookups = _make_hash_tables(articles_ds, customers_ds)
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
    return lookups, articles_tf

def make_tfds(
        tfrec_dir: str,
        config: Dict[str, Any],
        ts_len: int,
        mod_k: int=7,
        n_train: int=4,
        n_valid: int=5
        ):
    _f = lambda x, y: y
    _g1 = lambda x, y: x % mod_k <= n_train
    _g2 = lambda x, y: (n_train < x % mod_k) & (x % mod_k <= n_valid)
    _g3 = lambda x, y: (x % mod_k) > n_valid
    parse_f = lambda x: serialize.parse(x, ts_len=ts_len)
    tfrec_files = [os.path.join(tfrec_dir, f)
                   for f in os.listdir(tfrec_dir)]
    filenames = tf.data.Dataset.from_tensor_slices(tfrec_files)
    transactions_tf = (
        filenames
            .interleave(
                lambda filename: tf.data.TFRecordDataset(filename)
                    .map(parse_f, num_parallel_calls=2),
                cycle_length=4,
                num_parallel_calls=tf.data.AUTOTUNE))
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
