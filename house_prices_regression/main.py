"""
To pull dataset:
kaggle competitions download -c house-prices-advanced-regression-techniques
"""
import os
import sys
from typing import List, Dict, Tuple
import logging
import re
import time
import pandas as pd
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklmodsel
import sklearn
import sklearn.pipeline as sklpipe
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import load
import preprocessing
import modelzoo

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
SEED = 0
np.random.seed(seed=SEED)
logger.info(f"Random seed: {SEED}.")

default_results_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "results.csv"))

def run_models(models: Dict[int, Tuple[sklpipe.Pipeline]],
               df_train: pd.DataFrame,
               y_train: np.array,
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
    y_train = y_train.ravel()
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

def get_prediction(
        scores: pd.DataFrame,
        models: Dict[int, Tuple[sklpipe.Pipeline]],
        df_test: pd.DataFrame,
        id_test: np.array,
        prc_scl: sklpre.StandardScaler) -> None:
    idx = scores.iloc[0]["idx"]
    model = models[idx]
    pred = model.predict(df_test)
    pred = prc_scl.inverse_transform(pred.reshape((-1, 1)))
    pred = np.exp(pred.ravel())
    results = pd.DataFrame({"Id": id_test.ravel(), "SalePrice": pred,})
    return results

def main(outfn=default_results_fn):
    df_train, y_train, id_train, df_test, id_test = load.load_data()
    df_train, cat_map = preprocessing.encode_categorical(df_train)
    df_train, std_scl = preprocessing.scale_continuous(df_train)
    df_train, one_enc = preprocessing.encode_onehot(df_train)
    y_train, prc_scl = preprocessing.transform_price(y_train)
    dim_feats = len(df_train.columns)
    projectors = modelzoo.make_projectors(dim_feats)
    models = modelzoo.make_pipelines(projectors)
    scores = run_models(models, df_train, y_train)
    scores.to_csv("comparison.csv", index=False)
    df_test = preprocessing.transform(df_test, cat_map, std_scl, one_enc)
    results = get_prediction(scores, models, df_test, id_test, prc_scl)
    results.to_csv(outfn, index=False)
    return df_train, y_train, id_train, df_test, id_test, scores, results

if __name__=="__main__":
    main()
