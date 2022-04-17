import sys
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf

unk = "[unk]"

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

def make_tfds(
        transactions_ds: pd.DataFrame,
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame,
        ts_len: int=5):
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
    transactions_tf = tf.data.Dataset.from_tensor_slices(
        {
            "price": transactions_ds[[
                f"price_{n}" for n in range(ts_len, 0, -1)]].values,
            "sales_channel_id": transactions_ds[[
                f"sales_channel_id_{n}" for n in range(ts_len, 0, -1)]].values,
            "article_id_hist": transactions_ds[[
                f"article_id_{n}" for n in range(ts_len, 0, -1)]].values,
            "customer_id": transactions_ds["customer_id"].values,
            "article_id": transactions_ds["article_id"].values,
        })
    def _map(x):
        return {
            "customer_id": x["customer_id"],
            "product_type_name":
                lookups["product_type_name"].lookup(x["article_id_hist"]),
            "product_group_name":
                lookups["product_group_name"].lookup(x["article_id_hist"]),
            "graphical_appearance_name":
                lookups["graphical_appearance_name"].lookup(x["article_id_hist"]),
            "colour_group_name":
                lookups["colour_group_name"].lookup(x["article_id_hist"]),
            "perceived_colour_value_name":
                lookups["perceived_colour_value_name"].lookup(x["article_id_hist"]),
            "perceived_colour_master_name":
                lookups["perceived_colour_master_name"].lookup(x["article_id_hist"]),
            "department_name":
                lookups["department_name"].lookup(x["article_id_hist"]),
            "index_name":
                lookups["index_name"].lookup(x["article_id_hist"]),
            "index_group_name":
                lookups["index_group_name"].lookup(x["article_id_hist"]),
            "section_name":
                lookups["section_name"].lookup(x["article_id_hist"]),
            "garment_group_name":
                lookups["garment_group_name"].lookup(x["article_id_hist"]),
            "detail_desc":
                lookups["detail_desc"].lookup(x["article_id_hist"]),
            "age":
                lookups["age"].lookup(x["customer_id"]),
            "club_member_status":
                lookups["club_member_status"].lookup(x["customer_id"]),
            "fashion_news_frequency":
                lookups["fashion_news_frequency"].lookup(x["customer_id"]),
            "price": x["price"],
            "sales_channel_id": x["sales_channel_id"],
            "article_id": x["article_id"],}
    transactions_tf = transactions_tf.map(_map)
    return transactions_tf, articles_tf

