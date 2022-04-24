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
import sklearn.metrics
import sklearn.preprocessing
import nltk
import tensorflow as tf
import tensorflow_hub as hub
import official.nlp.bert
import official.nlp.bert.tokenization

SEED = 1
tf.random.set_seed(SEED)

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)
logging.captureWarnings(True)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.info(f"Python version: {sys.version}")
logger.info(f"Numpy version: {np.__version__}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info(f"Scikit-learn version: {sklearn.__version__}")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"tensorflow.random seed: {SEED}")

UNK = "UNK"
NUM = "number"
AT = "recipient"
http = "http"
html = "html"

target = "target"
keyword = "keyword"
location = "location"
hashtag = "hashtag"
at = "at"
href = "href"

y_cols = [f"{target}_0", f"{target}_1"]

def download_stopwords():
    try:
        nltk.download('stopwords')
    except:
        logger.error('...')
    try:
        stopwords = (nltk.corpus.stopwords.words("english")
                + ["u", "im", "st", "nd", "rd", "th"])
    except:
        logger.error('...')
    return stopwords

default_train_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, os.pardir, "data",
        "train.csv"))
default_test_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, os.pardir, "data",
        "test.csv"))
default_results_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, os.pardir, "data",
        "results.csv"))

def load_data(train_fn: str=default_train_fn,
              test_fn: str=default_test_fn,
              ) -> Tuple[pd.DataFrame]:
    df_train = pd.read_csv(train_fn)
    df_test = pd.read_csv(test_fn)
    id_train = df_train["id"]
    id_test = df_test["id"]
    y_train = df_train["target"]
    df_train.drop(["id", "target"], axis=1, inplace=True)
    df_test.drop("id", axis=1, inplace=True)
    return df_train, y_train, id_train, df_test, id_test

def prep(df: pd.DataFrame, stopwords: List[str]) -> pd.DataFrame:
    col = "text"
    df["text"] = df["text"].str.casefold()
    reg_hash_full = re.compile("(#)\w+")
    reg_hash = re.compile("(#)")
    reg_at = re.compile("(@)")
    reg_at_full = re.compile("(@)\w+")
    reg_href_full = re.compile("(htt)\S+")
    reg_html = re.compile("(&)\w+(;)")
    reg_x89 = re.compile(b"\xc2\x89".decode('utf-8')+"\S+")
    reg_special = re.compile("[^\w\s]")
    reg_contraction = re.compile("\s(s|m|t|(nt)|(ve)|w)\s")
    reg_numerals = re.compile("\d+[\s\d]*")
    reg_whitespace = re.compile("\s+")
    stop_str = ("(\s" + "\s)|(\s".join(stopwords) + "\s)"
            "(\A" + "\s)|(\A".join(stopwords) + "\s)"
            "(\s" + "\Z)|(\s".join(stopwords) + "\Z)")
    reg_stopwords = re.compile(stop_str)
    f = lambda x: [y.group() for y in reg_hash_full.finditer(x)]
    g = lambda x: ' '.join(x)
    df["hashtag"] = df["text"].apply(f).apply(g)
    df[col] = df[col].str.replace(reg_hash, ' ')
    f = lambda x: [y.group() for y in reg_at_full.finditer(x)]
    g = lambda x: ' '.join(x)
    df["at"] = df[col].apply(f).apply(g)
    df[col] = df[col].str.replace(reg_at_full, f" {AT} ")
    f = lambda x: len(list(reg_href_full.finditer(x)))
    df["href"] = df[col].apply(f)
    df[col] = df[col].str.replace(reg_href_full, f' {http} ')
    df[col] = df[col].str.replace(reg_html, f' {html} ')
    df[col] = df[col].str.replace(reg_x89, ' ')
    df[col] = df[col].str.replace(reg_special, ' ')
    df[col] = df[col].str.replace('_', ' ')
    df[col] = df[col].str.replace(reg_contraction, ' ')
    df[col] = df[col].str.replace(reg_numerals, f' {NUM} ')
    df[col] = df[col].str.replace(reg_stopwords, " ")
    df[col] = df[col].str.replace(reg_whitespace, " ")
    df[col] = df[col].str.strip()
    return df

def prep_tf_logging():
    tfboard_dir = "logs"
    if not os.path.exists(tfboard_dir):
        os.mkdir(tfboard_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tfboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        min_delta=1e-5,
        patience=10,
        baseline=0.5,
        restore_best_weights=True,)
    return [tensorboard_callback, early_stopping]

def make_dataset(input_dict: Dict[str, np.array],
                 outputs,
                 batch_size: int=128,
                 shards: int=10,
                 validation_k: int=3):
    x = tf.data.Dataset.from_tensor_slices(input_dict)
    ytups = []
    for inp in outputs:
        ytups.append(
            tf.data.Dataset.from_tensor_slices(inp))
    y = tf.data.Dataset.zip(tuple(ytups))
    x = tf.data.Dataset.zip((x, y))
    x_train = (x.enumerate()
                .filter(lambda x, y: x % shards > validation_k)
                .map(lambda x, y: y))
    x_valid = (x.enumerate()
                .filter(lambda x, y: x % shards <= validation_k)
                .map(lambda x, y: y))
    x_train = x_train.batch(batch_size)
    x_valid = x_valid.batch(batch_size)
    return x_train, x_valid

def transform_dataset(df: pd.DataFrame,
                      stopwords: List[str],
                      tokenizer,
                      sequence_length: int):
    df = prep(df, stopwords=stopwords)
    text_arr = df["text"].tolist()
    prewords = tokenizer.texts_to_sequences(text_arr)
    words_ids = tf.keras.preprocessing.sequence.pad_sequences(
                                                prewords,
                                                maxlen=sequence_length)
    premask = [[1 for i in arr] for arr in prewords]
    masks = tf.keras.preprocessing.sequence.pad_sequences(
                                                premask,
                                                maxlen=sequence_length)
    type_ids = np.zeros(words_ids.shape, dtype=np.int32)
    inputs = {
        "words_ids": words_ids,
        "masks": masks,
        "segment_ids": type_ids}
    return inputs

def predict_write(test_ds: Dict[str, np.array],
                  idxs: np.array,
                  model: tf.keras.Model,
                  results_fn: str=default_results_fn,
                  ):
    y_pred = model.predict(test_ds).ravel()
    results = pd.DataFrame({"id": idxs, "target": y_pred})
    results.to_csv(results_fn, index=False)

def main():
    df_train, y_train, id_train, df_test, id_test = load_data()
    stopwords = download_stopwords()
    df_train = prep(df_train, stopwords=stopwords)
    if False:
        tokenizer, num_unique_words, words_ids, masks, type_ids = (
                                bert_tokenize(df_train["text"].tolist()))
    else:
        tokenizer, num_unique_words, words_ids, masks, type_ids = (
                                tf_tokenizer(df_train["text"].tolist()))
    sequence_length = words_ids.shape[1]
    logger.info(f"Sequence length: {sequence_length}")
    logger.info(f"Vocab size: {num_unique_words}")
    inputs = {
        "words_ids": words_ids,
        "masks": masks,
        "segment_ids": type_ids}

    """
    bert_layer = load_pretrained_bert()
    model = build_bert_model(bert_layer,
                             sequence_length=sequence_length,
                             units_0=100,)
    """
    model = build_two_layer_model(
                             sequence_length=sequence_length,
                             num_unique_words=num_unique_words,
                             embed_dim=200,
                             units_0=100,
                             units_1=100,)
    """
    model = build_conv_model(sequence_length=sequence_length,
    				num_unique_words=num_unique_words,
				embed_dim=200,
				filters=5,
				window=5,
				pool_size=3,
				units=100,
				dense_0_dim=100,)
    """
    x_train, x_valid = make_dataset(
                                    inputs,
                                    [y_train],)
    hist = model.fit(x_train,
                     epochs=8,
                     validation_data=x_valid,)
    model.summary()
    x_test = transform_dataset(df_test,
                      stopwords,
                      tokenizer,
                      sequence_length)
    predict_write(x_test, id_test, model)

if __name__=="__main__":
    main()
