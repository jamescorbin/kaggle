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

#def make_test_dataset():
#    test_submissions = "data/sample_submission.csv"
#    transactions_parquet = "data/transactions.parquet"
#    customer_ids = pd.read_csv(test_submissions,
#                               usecols=["customer_id"])
#    test_data = pd.read_parquet(transactions_parquet)
#    test_data = test_data.loc[test_data["test"]==1]
#    customer_ids = (set(customer_ids["customer_id"].values.tolist())
#                  - set(test_data["customer_id"].values.tolist()))
#    customer_ids = pd.DataFrame({"customer_id": list(customer_ids)})
#    for col in test_data.columns:
#        if re.match(r"article_id", col):
#            customer_ids[col] = 0
#        elif re.match("sales_channel_id", col):
#            customer_ids[col] = 0
#        elif re.match("price_\d{1}_mask", col):
#            customer_ids[col] = False
#        elif re.match("price_\d{1}", col):
#            customer_ids[col] = 0.0
#    test_data = pd.concat([test_data, customer_ids],
#                          ignore_index=True,)
#    test_data.to_parquet("data/test.parquet")

def predict():
    config_fn = "./config-out.json"
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
