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
import model

def run_serialization(
        articles_fn: str,
        customers_fn: str,
        transactions_fn: str,
        tfrec_dir: str,
        vocab_dir: str,
        ) -> None:
    transactions_ds = rawdata.load_transactions_ds(transactions_fn)
    customers_ds = rawdata.load_customers_ds(customers_fn)
    articles_ds = rawdata.load_articles_ds(articles_fn)
    rawdata.write_vocabulary(articles_ds, customers_ds,
            parent_dir=vocab_dir)
    vocabulary = rawdata.load_vocabulary(
            parent_dir=vocab_dir)
    article_ids_map = {
        x: i
        for i, x in enumerate(vocabulary["article_ids"])}
    transactions_ds["target"] = (transactions_ds["article_id"]
                                 .apply(lambda x: article_ids_map[x]))
    serialize.write_dataset(
            transactions_ds,
            articles_ds,
            customers_ds)


if __name__=="__main__":
    filenames = [os.path.join("tfrec", x) for x in os.listdir("tfrec")]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(serialize.parse)
    model = model.RankModel(vocabulary, name="model_a")
    y_ds = dataset.map(lambda x: {"target": x["target"]})
    dataset = tf.data.Dataset.zip((dataset, y_ds))
    dataset = dataset.batch(64, drop_remainder=True)
    model.fit(dataset, epochs=2)
