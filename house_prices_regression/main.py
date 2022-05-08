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
import sklearn
import sklearn.preprocessing as sklpre
import sklearn.pipeline as sklpipe
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import extract
import transform
import train
import modelzoo

logging.captureWarnings(True)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler],)
logger = logging.getLogger(name=__name__)
logger.info(f"Python version: {sys.version}")
logger.info(f"Numpy version: {np.__version__}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info(f"Scikit-learn version: {sklearn.__version__}")
SEED = 0
np.random.seed(seed=SEED)
logger.info(f"Random seed: {SEED}.")

default_results_fn = os.path.abspath(os.path.join(
        __file__, os.pardir, "data", "results.csv"))

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
    df_train = extract.load_train_data()
    df_test = extract.load_test_data()
    df_train, cat_map = transform.encode_categorical(df_train)
    df_train, std_scl = transform.scale_continuous(df_train)
    df_train, one_enc = transform.encode_onehot(df_train)
    df_train, prc_scl = transform.transform_price(df_train)
    dim_feats = len(df_train.columns)
    projectors = modelzoo.make_projectors(dim_feats)
    models = modelzoo.make_pipelines(projectors)
    scores = train.run_models(models, df_train)
    scores.to_csv("comparison.csv", index=False)
    df_test = preprocessing.transform(df_test, cat_map, std_scl, one_enc)
    results = get_prediction(scores, models, df_test, id_test, prc_scl)
    results.to_csv(outfn, index=False)
    return df_train, y_train, id_train, df_test, id_test, scores, results

if __name__=="__main__":
    main()
