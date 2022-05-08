"""
kaggle competitions download -c tpu-getting-started
"""
import sys
import os
import math
import json
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import tensorflow_datasets as tfds
import extract
import imageaug
import load
import transform


def main():
    config_fn = "config-model.json"
    with open(config_fn, "r") as f:
        config = json.load(f)
    print(f"TF version: {tf.__version__}")
    dim = config["image_shape"][0]
    print(f"Image dimension: {dim}")
    print(f"Epochs initial: {config['epochs_init']}")
    print(f"Epochs tuning: {config['epochs_tune']}")

    ds = extract.get_training_dataset(config)
    print(ds)
    x_train, x_valid, x_test = load.get_dataset(ds, config)
    print(x_train)

if __name__=="__main__":
    main()
