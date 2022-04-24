import os
import sys
from typing import List, Dict, Tuple
import logging
import re
import time
import pandas as pd
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.pipeline as sklpipe
import sklearn.metrics
import sklearn.base as sklb
import sklearn.model_selection as sklmodsel
import sklearn.random_projection
import sklearn.decomposition
import sklearn.cluster
import sklearn.feature_selection
import xgboost
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.kernel_ridge
import sklearn.tree
import tensorflow as tf

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
        __file__, os.pardir, os.pardir, "data",
        "results.csv"))

def make_projectors(feat_dim: int,
                    ) -> List[Tuple[sklb.TransformerMixin, int]]:
    projs = [("passthrough", feat_dim),]
    n_components = [150]
    projs.extend([(sklearn.decomposition.PCA(
            n_components=n_comp,
            svd_solver="randomized",
            whiten=True,), n_comp)
            for n_comp in n_components])
    n_components = [150]
    projs.extend([
            (sklearn.cluster.FeatureAgglomeration(
                n_clusters=n_comp,), n_comp)
            for n_comp in n_components])
    return projs

def make_tfmodel(input_dim: int,
                units0: int=40,
                units1: int=5) -> tf.keras.Model:
    outdim = 1
    inputs = tf.keras.Input(shape=(input_dim),
                            dtype=tf.dtypes.float32)
    dense0 = tf.keras.layers.Dense(
                                units0,
                                activation=tf.nn.relu,
                                name="dense0")
    dense1 = tf.keras.layers.Dense(
                                units1,
                                activation=tf.nn.relu,
                                name="dense1",)
    densef = tf.keras.layers.Dense(
                                outdim,
                                activation="linear",
                                name="final",)
    opt = tf.keras.optimizers.Adam(
                            learning_rate=1e-4)
    x = dense0(inputs)
    x = dense1(x)
    x = densef(x)
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    loss = tf.keras.losses.MeanSquaredError()
    model = tf.keras.Model(inputs=[inputs],
                           outputs=[x],
                           name="tf_model")
    model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics,)
    return model

def make_pipelines(projs: List[Tuple[sklb.TransformerMixin, int]],
                   ) -> Dict[int, Tuple[sklpipe.Pipeline]]:
    models = {}
    param_grid = sklmodsel.ParameterGrid(
        dict(units0=[50, 500, 750],
            units1=[10, 100, 200],))
    model_counter = 0
    """
    for params in param_grid:
        for proj, dim in projs:
            models[model_counter] = sklearn.pipeline.Pipeline([
                ("projector", proj),
                ("model", make_tfmodel(input_dim=dim, **params))])
            model_counter += 1
    """
    grid_est = [
        (
            sklmodsel.ParameterGrid(dict()),
            sklearn.linear_model.LinearRegression),
        (
            sklmodsel.ParameterGrid(
                dict(C=[1, 2, 0.5],)),
            sklearn.svm.LinearSVR),
        (
            sklmodsel.ParameterGrid(dict(
                #n_estimators=[100, 110, 150],
                #max_depth=[5, 6, 8],
                learning_rate=[None, 1e-4, 1e-2],
                booster=["gbtree", "gblinear", "dart"],
                reg_alpha=[None, 1e-5, 1e-3],
                reg_lambda=[None, 1e-5, 1e-3],)),
            xgboost.XGBRegressor),
        (
            sklmodsel.ParameterGrid(
                dict(
                    C=[1, 2, 0.5],
                    kernel=["linear", "poly", "rbf", "sigmoid"],
                    gamma=["scale", 0.01],)),
            sklearn.svm.SVR),
        (
            sklmodsel.ParameterGrid(
                dict(alpha=[1, 2, 1e-1],
                    gamma=[None, 1, 0.1],)),
            sklearn.kernel_ridge.KernelRidge),
        (
            sklmodsel.ParameterGrid(
                dict(
                    alpha=[2, 1, 0.5],
                    l1_ratio=[0.5, 1, 0.1],)),
            sklearn.linear_model.ElasticNet),
        (
            sklmodsel.ParameterGrid(
                dict(kernel=[None,
                    sklearn.gaussian_process.kernels.Matern(),
                    sklearn.gaussian_process.kernels.Matern(nu=0.5),
                    sklearn.gaussian_process.kernels.Matern(nu=2.5),
                    sklearn.gaussian_process.kernels.Matern(nu=np.inf),
                    sklearn.gaussian_process.kernels.DotProduct(),
                    sklearn.gaussian_process.kernels.RationalQuadratic(),],)),
            sklearn.gaussian_process.GaussianProcessRegressor),
        (
            sklmodsel.ParameterGrid(
                dict(l1_ratio=[0.5, 0.1, 0.7, 0.9, 0.95, 1],)),
            sklearn.linear_model.ElasticNetCV),
        (
            sklmodsel.ParameterGrid(
                dict(n_estimators=[100],
                    criterion=["friedman_mse", "mse", "mae"],
                    max_depth=[3, 5],
                    max_features=[
                        "auto", "sqrt", "log2"],)),
            sklearn.ensemble.GradientBoostingRegressor),
        (
            sklmodsel.ParameterGrid(
                dict(n_estimators=[10, 20],
                    max_features=[1.0, 0.2],
                    bootstrap=[True, False],)),
            sklearn.ensemble.BaggingRegressor),
        (
            sklmodsel.ParameterGrid(
                dict(n_estimators=[100],
                    criterion=["mse", "mae"],
                    max_depth=[None, 5],
                    max_features=[
                        "auto",
                        "sqrt", "log2"],)),
            sklearn.ensemble.RandomForestRegressor),
        (
            sklmodsel.ParameterGrid(
                dict(alpha_1=[1e-6, 1e-5],
                    alpha_2=[1e-6, 1e-5],
                    lambda_1=[1e-6, 1e-5],
                    lambda_2=[1e-6, 1e-5],)),
            sklearn.linear_model.ARDRegression),
        (
            sklmodsel.ParameterGrid(
                dict(alpha=[1, 2, 0.5],)),
            sklearn.linear_model.Ridge),
        (
            sklmodsel.ParameterGrid(
                dict(eta0=[0.01, 0.005],
                    power_t=[0.25, 0.2],)),
            sklearn.linear_model.SGDRegressor),
        (
            sklmodsel.ParameterGrid(
                dict(base_estimator=[None,
                        sklearn.tree.DecisionTreeRegressor(max_depth=4),],
                    loss=["linear", "square", "exponential"],)),
            sklearn.ensemble.AdaBoostRegressor),
    ]
    for param_grid, est in grid_est:
        for params in param_grid:
            for proj, dim in projs:
                models[model_counter] = (
                    sklpipe.Pipeline([
                                    ("projector", proj),
                                ("model", est(**params))]))
                model_counter += 1
    return models

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
        scores["name"].append(str(model))
        scores["params"].append(model.get_params())
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
    df_train, y_train, id_train, df_test, id_test = load_data()
    df_train, cat_map = preprocessing.encode_categorical(df_train)
    df_train, std_scl = preprocessing.scale_continuous(df_train)
    df_train, one_enc = preprocessing.encode_onehot(df_train)
    y_train, prc_scl = preprocessing.transform_price(y_train)
    dim_feats = len(df_train.columns)
    projectors = make_projectors(dim_feats)
    models = make_pipelines(projectors)
    scores = run_models(models, df_train, y_train)
    df_test = preprocessing.transform(df_test, cat_map, std_scl, one_enc)
    results = get_prediction(scores, models, df_test, id_test, prc_scl)
    results.to_csv(outfn, index=False)
    return df_train, y_train, id_train, df_test, id_test, scores, results

if __name__=="__main__":
    main()
