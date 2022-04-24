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
import tensorflow_recommenders as tfrs

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

def main():
    import json
    with open("./config-model.json", "r") as f:
        config = json.load(f)
    vocab_dir = "vocabulary"
    tfrec_dir = "./tfrec"
    articles_fn = "./data/articles.csv"
    customers_fn = "./data/customers.csv"
    transactions_fn = "./data/transactions_train.csv"
    tfboard_log_dir = "./tfboard"
    model_save_pt = "model.hdf5"
    transactions_parquet = "./data/transactions.parquet"
    ts_len = 4
    #serialize.run_serialization(
    #        articles_fn,
    #        customers_fn,
    #        transactions_fn,
    #        tfrec_dir,
    #        vocab_dir,
    #        transactions_parquet=transactions_parquet,
    #        ts_len=ts_len)
    vocabulary = rawdata.load_vocabulary(parent_dir=vocab_dir)
    customers_ds = rawdata.load_customers_ds(customers_fn)
    articles_ds = rawdata.load_articles_ds(articles_fn)
    logger.info("Serializing dataset")
    articles_ds, customers_ds = rawdata.vectorize_features(
            articles_ds,
            customers_ds,
            vocabulary)
    lookups, articles_tf = tfsalesdata.make_articles_tf(
            articles_ds,
            customers_ds,)
    del articles_ds
    del customers_ds
    strategy = tf.distribute.get_strategy()
    batch_size = config["batch_size"] * strategy.num_replicas_in_sync
    with strategy.scope():
        xtrain, xvalid, xtest = tfsalesdata.make_tfds(
                tfrec_dir,
                config=config,
                ts_len=ts_len)
        model = recommendmodel.RetrievalModel(
                vocabulary,
                articles_tf,
                lookups,
                config,
                name="model_a")
        tfboard = tf.keras.callbacks.TensorBoard(
                log_dir=tfboard_log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=False,
                update_freq="epoch",
                profile_batch=0,
                embeddings_freq=1,
                embeddings_metadata=None,)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                model_save_pt,
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                save_freq="epoch",
                options=None,
                initial_value_threshold=None,)
        early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=0,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True)
        callbacks = [tfboard, model_checkpoint, early_stopping]
        model.fit(
                xtrain
                    .batch(config["batch_size"], drop_remainder=True)
                    .take(10)
                    .prefetch(tf.data.AUTOTUNE),
                validation_data=xvalid
                    .batch(config["batch_size"])
                    .take(10)
                    .prefetch(tf.data.AUTOTUNE),
                epochs=config["epochs"],
                callbacks=callbacks)
        #model.evaluate(
        #        xtrain.batch(config["batch_size"]),
        #        callbacks=callbacks)
        #model.evaluate(
        #        xvalid.batch(config["batch_size"]),
        #        callbacks=callbacks)
        #model.evaluate(
        #        xtest.batch(config["batch_size"]),
        #        callbacks=callbacks)
    test_data = rawdata.get_test_data(
            pd.read_parquet(transactions_parquet),
            ts_len=ts_len)
    index = tfrs.layers.factorized_top_k.BruteForce(
            model.customer_model)
    index.index_from_dataset(
        articles_tf.batch(1024).map(lambda x:
            (x["article_id"], model.article_model(x))))
    titles = test_data.batch(1024).map(lambda x: index(x, k=12)[1])
    prediction = []
    for title in titles:
        prediction.append(np.array(title))
    prediction = pd.DataFrame(np.concatenate(prediction, axis=0))
    prediction.rename(
            {x: str(x) for x in prediction.columns},
            axis=1, inplace=True)
    prediction.to_parquet("prediction.parquet")

if __name__=="__main__":
    main()
