import sys
import os
import json
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import digitmodel
import extract
import load
import train
import mlflow

logger = logging.getLogger(name=__name__)

def decode_predictions(predictions: np.array) -> np.array:
    return np.argmax(predictions, axis=1).flatten()

if __name__=="__main__":
    config_fn = "config-model.json"
    tfboard_log_dir = "data/tfboard"
    model_save_pt = "data/model"
    model_checkpoint_pt = "data/model_checkpoint"
    with open(config_fn, "r") as f:
        config = json.load(f)
    tf.random.set_seed(config["seed"])
    train_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "train.csv"))
    x_train, y_train = extract.load_train_data(
            train_fn,
            image_shape=config["image_shape"])
    test_fn = "./data/test.csv"
    x_pred = extract.load_test_data(
            test_fn,
            image_shape=config["image_shape"])
    mlflow.set_tracking_uri("file:///home/jec/Desktop/artifacts")
    mlflow.set_experiment("digit_classifier")
    mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_model_signatures=True)
    with mlflow.start_run():
        model, hist, eval_test = train.run_training_loop(
            x_train,
            y_train,
            config)
        mlflow.log_params(config)
        mlflow.log_metrics({f"test_{key}": val
                            for key, val in eval_test.items()})
    y_pred = model.predict(x_pred)
    y_pred = decode_predictions(y_pred)
    results = pd.DataFrame(
        {"ImageId": np.arange(1, y_pred.shape[0] + 1),
               "Label": y_pred.flatten()})
    results.to_csv("results.csv", index=False)
