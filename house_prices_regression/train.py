import sys
import os
from typing import Dict, Tuple
import pandas as pd
import sklearn.pipeline as sklpipe
import sklearn
import sklearn.model_selection as sklmodsel
import time
import logging

logger = logging.getLogger(name=__name__)

def run_models(
        models: Dict[int, Tuple[sklpipe.Pipeline]],
        df_train: pd.DataFrame,
        ) -> pd.DataFrame:
    scores =  {
        "idx": [],
        "name": [],
        "params": [],
        "r2": [],
        "mse": [],
        "test_mse": [],
        "train_time": []}
    pred_cols = [f"model_{i:04d}" for i in range(len(models))]
    y_train = df_train[["SalePrice"]].values.ravel()
    for model_idx, model in models.items():
        t1 = time.perf_counter()
        X_train_split, X_test_split, y_train_split, y_test_split = (
            sklmodsel.train_test_split(
                df_train,
                y_train,
                test_size=0.2,))
        validation_data = (X_test_split, y_test_split)
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(df_train).ravel()
        r2 = sklearn.metrics.r2_score(y_train,
                                      y_pred)
        mse = sklearn.metrics.mean_squared_error(
                                    y_train, y_pred)
        test_mse = sklearn.metrics.mean_squared_error(
            y_test_split,
            model.predict(X_test_split).ravel(),)
        t2 = time.perf_counter()
        logger.info(f"{str(model)[:15]} -- time elapsed: {t2-t1:5.3f}")
        scores["idx"].append(model_idx)
        scores["name"].append(" ".join(str(model).split("\n")))
        scores["params"].append(" ".join(str(model.get_params()).split("\n")))
        scores["r2"].append(r2)
        scores["mse"].append(mse)
        scores["test_mse"].append(test_mse)
        scores["train_time"].append(t2-t1)
    scores = pd.DataFrame(scores)
    scores.sort_values(by=["test_mse"], ascending=True, inplace=True)
    return scores
