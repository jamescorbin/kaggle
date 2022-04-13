"""
kaggle competitions downloads -c h-and-m-personalized-fashion-recommendations
"""
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import serialize
import rawdata

def transform_raw_data_to_series(
            transactions_ds: pd.DataFrame,
            articles_ds: pd.DataFrame,
            customers_ds: pd.DataFrame,
            ts_len: int=5):
    dataset = []
    transactions_ds["t_dat"] = transactions_ds["t_dat"].str.encode("utf-8")
    transactions_ds["customer_id"] = (
            transactions_ds["customer_id"].str.encode("utf-8"))
    transactions_ds = transactions_ds.join(
            customers_ds,
            on="customer_id",
            how="left")
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
    for i, row in transactions_ds.iterrows():
        data = {}
        data["article_id"] = [
                row[f"article_id_{n}"] for n in range(ts_len, 0, -1)]
        for col in article_columns:
            data[col] = [row[f"{col}_{n}"] for n in range(ts_len, 0, -1)]
        data["t_dat"] = [row["t_dat"]]
        data["price"] = [row[f"price_{n}"]
                for n in range(ts_len, 0, -1)]
        data["sales_channel_id"] = [row[f"sales_channel_id_{n}"]
                for n in range(ts_len, 0, -1)]
        data["target"] = [row["article_id"]]
        cust_idx = row["customer_id"]
        data["customer_id"] = [cust_idx]
        data["FN"] = [row["FN"]]
        data["Active"] = [row["Active"]]
        data["club_member_status"] = [row["club_member_status"]]
        data["fashion_news_frequency"] = [row["fashion_news_frequency"]]
        data["age"] = [row["age"]]
        dataset.append(data)
    return dataset

if __name__=="__main__":
    articles_ds = rawdata.load_articles_ds()
    customers_ds = rawdata.load_customers_ds()
    transactions_ds = rawdata.load_transactions_ds()
    dataset = transform_raw_data_to_series(
            transactions_ds,
            articles_ds,
            customers_ds)
    out_fp = "test.tfrec"
    with tf.io.TFRecordWriter(out_fp) as writer:
        for data in dataset:
            writer.write(serialize.serialize_example(data))
    filenames = [out_fp]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(serialize.parse)
    d = dataset.take(10)
    for k in d:
        print(k)

