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
        tfrec_dir: str,
        vocab_dir: str,
        ) -> None:
    serialize.write_dataset(
            transactions_fn,
            tfrec_dir=tfrec_dir)
    customers_ds = rawdata.load_customers_ds(customers_fn)
    articles_ds = rawdata.load_articles_ds(articles_fn)
    rawdata.write_vocabulary(
            articles_ds,
            customers_ds,
            parent_dir=vocab_dir)

def load_data(
        articles_fn: str,
        customers_fn: str,
        tfrec_dir: str,
        vocab_dir: str,):
    vocabulary = rawdata.load_vocabulary(parent_dir=vocab_dir)
    customers_ds = rawdata.load_customers_ds(customers_fn)
    articles_ds = rawdata.load_articles_ds(articles_fn)
    logger.info("Serializing dataset")
    dataset, articles_tf = tfsalesdata.make_tfds(
            tfrec_dir,
            articles_ds,
            customers_ds)
    return dataset, articles_tf, vocabulary

if __name__=="__main__":
    import json
    with open("./model_cfg.json", "r") as f:
        config = json.load(f)
    vocab_dir = "vocabulary"
    tfrec_dir = "./tfrec"
    articles_fn = "./data/articles.csv"
    customers_fn = "./data/customers.csv"
    transactions_fn = "./data/transactions_train.csv"
    run_serialization(
            articles_fn,
            customers_fn,
            transactions_fn,
            tfrec_dir,
            vocab_dir)
    dataset, articles_tf, vocabulary = load_data(
            articles_fn,
            customers_fn,
            tfrec_dir,
            vocab_dir)
    model = model.RetrievalModel(vocabulary,
                                 articles_tf,
                                 config,
                                 name="model_a")
    dataset = dataset.batch(
            config["batch_size"],
            drop_remainder=True)
    model.fit(dataset, epochs=config["epochs"])
