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


if __name__=="__main__":
    articles_ds = rawdata.load_articles_ds()
    customers_ds = rawdata.load_customers_ds()
    transactions_ds = rawdata.load_transactions_ds()
    #rawdata.write_vocabulary(articles_ds, customers_ds)
    vocabulary = rawdata.load_vocabulary()
    article_ids_map = {
        x: i
        for i, x in enumerate(vocabulary["article_ids"])}
    print(article_ids_map)
    transactions_ds["target"] = (transactions_ds["article_id"]
                                 .apply(lambda x: article_ids_map[x]))
    print(transactions_ds["target"])
    dataset = serialize.write_dataset(
            transactions_ds,
            articles_ds,
            customers_ds)
    filenames = [os.path.join("tfrec", x) for x in os.listdir("tfrec")]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(serialize.parse)
    model = model.RankModel(vocabulary, name="model_a")
    y_ds = dataset.map(lambda x: {"target": x["target"]})
    dataset = tf.data.Dataset.zip((dataset, y_ds))
    dataset = dataset.batch(32, drop_remainder=True)
    model.fit(dataset, epochs=1)


