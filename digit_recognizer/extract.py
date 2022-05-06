import sys
import os
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import digitmodel
import logging

logger = logging.getLogger(name=__name__)

def load_train_data(fn: str,
                    image_shape: Tuple[int, int, int],
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(fn, header=0)
    x_train = (df.drop("label", axis=1)
               .values
               .reshape((len(df), image_shape[0],
                         image_shape[1], image_shape[2]))
               .astype(np.float32))
    y_train = pd.DataFrame(df[["label"]]).values
    x_train = convert_gray_to_float(x_train)
    return x_train, y_train

def load_test_data(fn: str,
                   image_shape: Tuple[int, int, int],
                   ) -> pd.DataFrame:
    df = pd.read_csv(fn, header=0)
    x_test = (df
               .values
               .reshape((len(df), image_shape[0],
                         image_shape[1], image_shape[2]))
               .astype(np.float32))
    x_test = convert_gray_to_float(x_test)
    return x_test

def convert_gray_to_float(images: np.array) -> np.array:
    images = (images -  128) / 255.0
    return images.astype(np.float32)
