"""
To pull dataset:
    kaggle competitions downloads -c h-and-m-personalized-fashion-recommendations
"""
import sys
import os
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import serialize
import rawdata
import recommendmodel
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
        vocab_dir: str,
        config: Dict[str, Any]):
    vocabulary = rawdata.load_vocabulary(parent_dir=vocab_dir)
    customers_ds = rawdata.load_customers_ds(customers_fn)
    articles_ds = rawdata.load_articles_ds(articles_fn)
    logger.info("Serializing dataset")
    lookups, articles_tf = tfsalesdata.make_articles_tf(
            articles_ds,
            customers_ds)
    dataset, articles_tf = tfsalesdata.make_tfds(
            tfrec_dir,
            lookups,
            articles_tf,
            config)
    return dataset, articles_tf, vocabulary, lookups

if __name__=="__main__":
    import json
    with open("./model_cfg.json", "r") as f:
        config = json.load(f)
    vocab_dir = "vocabulary"
    tfrec_dir = "./tfrec"
    articles_fn = "./data/articles.csv"
    customers_fn = "./data/customers.csv"
    transactions_fn = "./data/transactions_train.csv"
    """
    run_serialization(
            articles_fn,
            customers_fn,
            transactions_fn,
            tfrec_dir,
            vocab_dir)
    """
    dataset, articles_tf, vocabulary, lookups = load_data(
            articles_fn,
            customers_fn,
            tfrec_dir,
            vocab_dir,
            config)
    model = recommendmodel.RetrievalModel(
                                vocabulary,
                                articles_tf,
                                lookups,
                                config,
                                name="model_a")
    model.fit(dataset, epochs=config["epochs"])
