"""
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
"""
import os
import sys
import logging
import json
import itertools
from typing import List, Tuple, Optional, Dict
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import official.nlp.bert
import official.nlp.bert.tokenization
import tensorflow_text as tf_text
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import extract
import tweetmodel
import transform
import load

def main(config):
    df_train = extract.load_train_data()
    stopwords = extract.download_stopwords()
    #df_train = transform.prepare_sentences(df_train, stopwords=stopwords)
    #transform.write_vocabulary(df_train)

    ds = tf.data.Dataset.from_tensor_slices(
            {"text": df_train[["text"]].values})
    ds_y = tf.data.Dataset.from_tensor_slices(
            {"target": df_train[["target"]].values})
    ds = ds.take(1_000)
    ds = load.encode_dataset(ds, config)
    for k in ds.take(10):
        print(k)
    #model = tweetmodel.build_bert_model(config)
    model = tweetmodel.TweetModel(config)
    ds = tf.data.Dataset.zip((ds, ds_y)).batch(32, drop_remainder=True)
    model.fit(ds,
            epochs=1,)



if __name__=="__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    main(config)
