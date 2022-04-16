"""
To pull dataset:
    kaggle competitions downloads -c h-and-m-personalized-fashion-recommendations
"""
import sys
import os
import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import serialize
import rawdata
import model
import tfsalesdata

frmt = ("[%(asctime)s] %(levelname)s "
        "[%(name)s.%(funcName)s:%(lineno)d] "
        "%(message)s")
formatter = logging.Formatter(fmt=frmt)
stdout = logging.StreamHandler(stream=sys.stdout)
stdout.setFormatter(formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[stdout],)
logger = logging.getLogger(name=__name__)

def run_serialization(
        articles_fn: str,
        customers_fn: str,
        transactions_fn: str,
        vocab_dir: str,
        ) -> None:
    transactions_ds = rawdata.load_transactions_ds(transactions_fn)
    customers_ds = rawdata.load_customers_ds(customers_fn)
    articles_ds = rawdata.load_articles_ds(articles_fn)
    rawdata.write_vocabulary(articles_ds,
            customers_ds,
            parent_dir=vocab_dir)
    vocabulary = rawdata.load_vocabulary(parent_dir=vocab_dir)
    article_ids_map = {
            x: i for i, x in enumerate(vocabulary["article_id"])}
    transactions_ds["target"] = (
            transactions_ds["article_id"].apply(lambda x: article_ids_map[x]))
    logger.info("Serializing dataset")
    dataset = tfsalesdata.make_tfds(transactions_ds, articles_ds, customers_ds)
    return dataset

if __name__=="__main__":
    vocab_dir = "vocabulary"
    dataset = run_serialization(
            "./data/articles.csv",
            "./data/customers.csv",
            "./data/transactions_train.csv",
            vocab_dir
            )
    vocabulary = rawdata.load_vocabulary(parent_dir=vocab_dir)
    model = model.RankModel(vocabulary, name="model_a")
    dataset = dataset.batch(64, drop_remainder=True)
    model.fit(dataset, epochs=2)
