import os
import sys
from typing import List, Dict, Tuple
import logging
import re
import time
import pandas as pd
import numpy as np

default_train_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "train.csv"))
default_test_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "test.csv"))

def load_train_data(
        train_fn: str=default_train_fn,
        ) -> pd.DataFrame:
    df_train = pd.read_csv(train_fn)
    df_train = pd.DataFrame(
            df_train.loc[pd.notnull(df_train["SalePrice"])])
    return df_train

def load_test_data(
        test_fn: str=default_test_fn,
        ) -> pd.DataFrame:
    df_test = pd.read_csv(test_fn)
    return df_test
