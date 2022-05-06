import sys
import os
import pandas as pd
import nltk
from typing import Tuple

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
        __file__, os.pardir, "data", "train.csv"))
default_test_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "test.csv"))
default_results_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "results.csv"))

def load_train_data(
        train_fn: str=default_train_fn,
        ) -> Tuple[pd.DataFrame]:
    df_train = pd.read_csv(train_fn)
    return df_train

def load_test_data(
        test_fn: str=default_test_fn,
        ) -> pd.DataFrame:
    df_test = pd.read_csv(test_fn)
    return df_test
