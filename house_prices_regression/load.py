import os
import sys
from typing import List, Dict, Tuple
import logging
import re
import time
import pandas as pd
import numpy as np

default_train_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, os.pardir, "data",
        "train.csv"))
default_test_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, os.pardir, "data",
        "test.csv"))

def load_data(train_fn: str=default_train_fn,
              test_fn: str=default_test_fn,
              ) -> Tuple[pd.DataFrame]:
    df_train = pd.read_csv(train_fn)
    df_test = pd.read_csv(test_fn)
    df_train = pd.DataFrame(
            df_train.loc[pd.notnull(df_train["SalePrice"])])
    id_train = df_train["Id"]
    id_test = df_test["Id"]
    y_train = df_train[["SalePrice"]]
    df_train.drop("Id", axis=1, inplace=True)
    df_test.drop("Id", axis=1, inplace=True)
    df_train.drop("SalePrice", axis=1, inplace=True)
    return df_train, y_train, id_train, df_test, id_test
