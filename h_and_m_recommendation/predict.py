import sys
import os
import json
import re
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import pandas as pd
import numpy as np
import tensorflow as tf
import recommendmodel
import tfsalesdata
import rawdata
import tensorflow_recommenders as tfrs

def predict():
    config_fn = "config-model.json"
    with open(config_fn, "r") as f:
        config = json.load(f)
    tfrec_dir = "./data/tfrec"
    transactions_ds = tfsalesdata.get_prediction_data(
            tfrec_dir,
            config)
    transactions_ds = transactions_ds.take(2_000)
    article_model_pt = "./data/articles_model"
    article_model = tf.keras.models.load_model(article_model_pt)
    customer_model_pt = "./data/customer_model"
    customer_model = tf.keras.models.load_model(customer_model_pt)
    index = tfrs.layers.factorized_top_k.BruteForce(
            customer_model)
    articles_tf = tfsalesdata.make_articles_tf(
            rawdata.load_articles_ds(config["articles_fn"]))
    _f = lambda x: (x["article_id"], article_model(x))
    index.index_from_dataset(articles_tf.batch(1024).map(_f))
    _g = lambda x: index(x, k=12)[1]
    titles = transactions_ds.batch(1024).map(_g)
    prediction = []
    for title in titles:
        prediction.append(np.array(title))
    prediction = pd.DataFrame(np.concatenate(prediction, axis=0))
    prediction.rename(
            {x: str(x) for x in prediction.columns},
            axis=1, inplace=True)
    prediction.to_parquet("./data/prediction.parquet")

if __name__=="__main__":
    predict()
