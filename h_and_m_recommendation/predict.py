import sys
import os
import json
import re
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import pandas as pd
import numpy as np
import recommendmodel
import tfsalesdata
import rawdata
import tensorflow_recommenders as tfrs

def predict():
    config_fn = "config-model.json"
    with open(config_fn, "r") as f:
        config = json.load(f)
    transactions_parquet = "data/transactions.parquet"
    model_checkpoint_pt = "model-checkpoint"
    test_data = pd.read_parquet("data/test.parquet")
    test_data = rawdata.get_test_data(
            test_data,
            ts_len=config["ts_len"])
    model = recommendmodel.RetrievalModel(
            config,
            name="model_new")
    model.load_weights(model_checkpoint_pt)
    index = tfrs.layers.factorized_top_k.BruteForce(
            model.customer_model)
    articles_tf = tfsalesdata.make_articles_tf(
            rawdata.load_articles_ds(config["articles_fn"]))
    _f = lambda x: (x["article_id"], model.article_model(x))
    index.index_from_dataset(articles_tf.batch(1024).map(_f))
    _g = lambda x: index(x, k=12)[1]
    titles = test_data.batch(1024).map(_g)
    prediction = []
    for title in titles:
        prediction.append(np.array(title))
    prediction = pd.DataFrame(np.concatenate(prediction, axis=0))
    prediction.rename(
            {x: str(x) for x in prediction.columns},
            axis=1, inplace=True)
    prediction.to_parquet("prediction.parquet")

if __name__=="__main__":
    #make_test_dataset()
    predict()
