import sys
import os
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import tensorflow as tf
logger = logging.getLogger(name=__name__)
unk = "[UNK]"

default_articles_fn = os.path.abspath(os.path.join(
        __file__, os.pardir,
        "data", "articles.csv"))
default_articles_parquet = os.path.abspath(os.path.join(
        __file__, os.pardir,
        "data", "articles.parquet"))
default_customer_fn = os.path.abspath(os.path.join(
        __file__, os.pardir,
        "data", "customers.csv"))
default_transact_fn = os.path.abspath(os.path.join(
        __file__, os.pardir,
        "data", "transactions_train.csv"))

def load_articles_ds(
        articles_fn: str=default_articles_fn,
        articles_parquet: str=default_articles_parquet,
        ) -> pd.DataFrame:
    if not os.path.exists(articles_parquet):
        logger.info(f"Opening articles dataset")
        articles_bytes_cols = [
                "prod_name",
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
                "detail_desc",]
        articles_ds = pd.read_csv(articles_fn)
        articles_ds.drop(
                articles_ds.index[
                    pd.isnull(articles_ds["article_id"])],
                axis=0,
                inplace=True)
        for col in articles_bytes_cols:
            articles_ds[col] = articles_ds[col].fillna(unk)
        articles_ds.to_parquet(articles_parquet)
    else:
        articles_ds.read_parquet(articles_parquet)
    return articles_ds

def load_customers_ds(
        customers_fn: str=default_customer_fn,
        customers_parquet: str=default_customers_parquet,
        ) -> pd.DataFrame:
    if not os.path.exists(customers_parquet):
        logger.info(f"Opening customers dataset")
        customers_bytes_cols = [
                "customer_id",
                "club_member_status",
                "fashion_news_frequency",]
        customers_ds = pd.read_csv(customers_fn)
        customers_ds["fashion_news_frequency"] = (
                customers_ds["fashion_news_frequency"]
                    .fillna(unk))
        customers_ds["club_member_status"] = (
                customers_ds["club_member_status"]
                    .fillna(unk))
        customers_ds.loc["age_mask"] = (
                pd.notnull(customers_ds["age"]).astype(np.float32))
        m = customers_ds["age"].mean()
        std = customers_ds["age"].std()
        n = len(customers_ds)
        customers_ds["age"] = (customers_ds["age"]
                .fillna(pd.Series(np.random.random(m, std, n))))
        customers_ds.drop(
                customers_ds.index[
                    pd.isnull(customers_ds["customer_id"])],
                axis=0,
                inplace=True)
        customers_ds.to_parquet(customers_parquet)
    else:
        customers_ds.read_parquet(customers_parquet)
    return customers_ds

def convert_transaction_to_datapoint(
            row,
            ts_len: int) -> Dict[str, Any]:
    data = {}
    data["article_id_hist"] = [
            row[f"article_id_{n}"]
            for n in range(ts_len, 0, -1)]
    data["article_id"] = [row["article_id"]]
    data["t_dat"] = [row["t_dat"].encode("utf-8")]
    data["price"] = [row[f"price_{n}"]
            for n in range(ts_len, 0, -1)]
    data["price_mask"] = [row[f"price_{n}_mask"]
            for n in range(ts_len, 0, -1)]
    data["sales_channel_id"] = [
            row[f"sales_channel_id_{n}"]
            for n in range(ts_len, 0, -1)]
    data["customer_id"] = [row["customer_id"].encode("utf-8")]
    return data

def get_test_data(
            test_ds: pd.DataFrame,
            ts_len: int,
            ) -> tf.data.Dataset:
    data = {}
    data["article_id_hist"] = (
        test_ds[[f"article_id_{n}"
            for n in range(ts_len, 0, -1)]].values)
    data["article_id"] = test_ds[["article_id"]].values
    data["price"] = (
        test_ds[[f"price_{n}"
            for n in range(ts_len, 0, -1)]].values)
    data["price_mask"] = (
        test_ds[[f"price_{n}_mask"
            for n in range(ts_len, 0, -1)]].values)
    data["sales_channel_id"] = (
        test_ds[[f"sales_channel_id_{n}"
            for n in range(ts_len, 0, -1)]].values)
    data["customer_id"] = test_ds[["customer_id"]].values
    data = tf.data.Dataset.from_tensor_slices(data)
    return data

def append_previous_purchases(
        transaction_ds: pd.DataFrame,
        ts_len: int=5,
        ) -> pd.DataFrame:
    article_id = "article_id"
    price = "price"
    sales_channel_id = "sales_channel_id"
    article_id_fillna = -1
    for n in range(1, ts_len + 1):
        transaction_ds[f"{article_id}_{n}"] = (
                transaction_ds
                    .groupby("customer_id")
                    [article_id]
                    .shift(n)
                    .fillna(article_id_fillna)
                    .astype(int))
        transaction_ds[f"{price}_{n}"] = (
                transaction_ds
                    .groupby("customer_id")
                    [price]
                    .shift(n))
        transaction_ds[f"{sales_channel_id}_{n}"] = (
                transaction_ds
                    .groupby("customer_id")
                    [sales_channel_id]
                    .shift(n)
                    .fillna(0)
                    .astype(int))
    m, std = transactions_ds["price"].mean(), transactions_ds["price"].std()
    for n in range(1, ts_len + 1):
        transaction_ds[f"{price}_{n}_mask"] = (
                pd.notnull(transaction_ds[f"{price}_{n}"])
                    .astype(np.float32))
        transaction_ds[f"{price}_{n}"] = (transaction_ds[f"{price}_{n}"]
                .fillna(pd.Series(
                    np.random.normal(m, std, len(transaction_ds)))))
    return transaction_ds

def convert_transactions_csv(
        out_fp : str,
        transactions_fn: str=default_transact_fn,
        ts_len: int=3,
        ):
    if not os.path.exists(out_fp):
        test_submissions_fn = "data/sample_submission.csv"
        unique_customer_id = (
            pd.read_csv(test_submission_fn, usecols=["customer_id"])
            ["customer_id"])
        names = ["t_dat",
                 "customer_id",
                 "article_id",
                 "price",
                 "sales_channel_id"]
        transactions_ds = pd.read_csv(
                transactions_fn,
                skiprows=1,
                names=names,)
        transactions_ds["article_id"] = (
                transactions_ds["article_id"].astype(int))
        transactions_ds["test"] = 0
        test_set = pd.DataFrame(
            {
                "customer_id": unique_customer_id,
                "article_id": [-1] * len(unique_customer_id),
                "test": [1] * len(unique_customer_id)
            })
        transactions_ds = pd.concat([transactions_ds, test_set],
                                    axis=0,
                                    ignore_index=True)
        transactions_ds["sales_channel_id"] = (
                transactions_ds["sales_channel_id"]
                    .fillna(0).astype(int))
        transactions_ds = append_previous_purchases(
                transactions_ds,
                ts_len=ts_len)
        transactions_ds["t_dat"] = (
                transactions_ds["t_dat"]
                    .fillna("2020-09-23"))
        transactions_ds["t_dat"] = pd.to_datetime(
                transactions_ds["t_dat"])
        transactions_ds.to_parquet(out_fp)

def write_vocabulary(
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame,
        parent_dir: str="./vocabulary"):
    """
    """
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    article_cols = [
            "article_id",
            "prod_name",
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
            "garment_group_name",]
    customer_cols = [
            "customer_id",
            "club_member_status",
            "fashion_news_frequency"]
    vocabulary = {}
    for col in article_cols:
        vocabulary[col] = articles_ds[col].unique().tolist()
    for col in customer_cols:
        vocabulary[col] = customers_ds[col].unique().tolist()
    for col, lst in vocabulary.items():
        fn = os.path.join(parent_dir, f"{col}.txt")
        with open(fn, "w") as f:
            logger.info(os.path.basename(fn))
            for w in lst:
                if w not in (unk, -1):
                    try:
                        f.write(f"{w}\n")
                    except TypeError as e:
                        logger.error(e)
                        logger.error(w)
                        raise TypeError(e)
    return vocabulary

def load_vocabulary(
        config: Dict[str, Any],
        parent_dir: str="vocabulary",
        ) -> Dict[str, List[Any]]:
    """
    """
    vocabulary = {}
    cols = [
            "article_id",
            "prod_name",
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
            "customer_id",
            "club_member_status",
            "fashion_news_frequency"]
    for col in cols:
        fn = os.path.join(parent_dir, f"{col}.txt")
        with open(fn, "r") as f:
            vocabulary[col] = [w.strip() for w in f]
        config[f"{col}_dim"] = len(vocabulary[col]) + 1
        config[f"{col}"] = fn
    return vocabulary
