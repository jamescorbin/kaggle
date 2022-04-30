"""
To pull dataset:
    kaggle competitions downloads -c h-and-m-personalized-fashion-recommendations
"""
import sys
import os
import logging
import json
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

def run_training_loop(
        config_fn: str,
        tfrec_dir: str,
        tfboard_log_dir: str,
        model_save_pt: str,
        articles_model_save: str,
        customer_model_save: str,
        ):
    with open(config_fn, "r") as f:
        config = json.load(f)
    ts_len = config["ts_len"]
    tf.random.set_seed(config["seed"])
    strategy = tf.distribute.get_strategy()
    batch_size = config["batch_size"] * strategy.num_replicas_in_sync
    with strategy.scope():
        xtrain, xvalid, xtest = tfsalesdata.make_tfds(
                tfrec_dir,
                config=config,
                ts_len=ts_len)
        #xtrain, xvalid, xtest = (xtrain.take(1_000), 
        #       xvalid.take(5_000), xtest.take(1_000))
        model = recommendmodel.RetrievalModel(
                config,
                name="model_5")
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
                patience=2,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True)
        callbacks = [tfboard, model_checkpoint, early_stopping]
        model.fit(
                xtrain
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE),
                validation_data=xvalid
                    .batch(2**13)
                    .prefetch(tf.data.AUTOTUNE),
                epochs=config["epochs"],
                callbacks=callbacks)
        test_eval = model.evaluate(
                xtest.batch(2**13),
                callbacks=callbacks)
    model.customer_model.save(customer_model_save)
    model.article_model.save(articles_model_save)

if __name__=="__main__":
    config_fn = "config-model.json"
    tfrec_dir = "data/tfrec"
    tfboard_log_dir = "data/tfboard"
    model_save_pt = "data/model.hdf5"
    articles_model_save = "data/articles_model"
    customer_model_save = "data/customer_model"
    run_training_loop(
        config_fn,
        tfrec_dir,
        tfboard_log_dir,
        model_save_pt,
        articles_model_save,
        customer_model_save)
