import os
import sys
import math
from typing import List, Tuple
import pandas as pd
import re
import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.metrics
import sklearn.preprocessing as sklpre

SEED = 0
np.random.seed(seed=SEED)

default_train_fn = os.path.abspath(os.path.join(
    __file__, os.pardir, os.pardir, "data", "train.csv"))
default_test_fn = os.path.abspath(os.path.join(
    __file__, os.pardir, os.pardir, "data", "test.csv"))
default_results_fn = os.path.abspath(os.path.join(
    __file__, os.pardir, os.pardir, "data", "results.csv"))

title_regex = re.compile("\w+\s?\w*(\.)")
cont_feats = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",]

cat_feats = [
    "Pclass",
    "Sex",
    "Embarked",
    "Name",
    "Cabin",
    "Ticket",
    "Ticket_is_digits",]

def load_data(train_fn: str=default_train_fn,
              test_fn: str=default_test_fn,
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(train_fn)
    idx_train = df_train["PassengerId"]
    y_train = df_train["Survived"]
    df_train.drop(["PassengerId", "Survived"],
                  axis=1, inplace=True)
    df_test = pd.read_csv(test_fn)
    idx_test = df_test["PassengerId"]
    df_test.drop(["PassengerId"],
                  axis=1, inplace=True)
    return (idx_train, df_train, y_train, idx_test, df_test)

def apply_feature_transform(df: pd.DataFrame) -> pd.DataFrame:
    df["Name"] = (
        df["Name"]
            .apply(lambda x: x.split(',')[1].strip())
            .apply(lambda x: title_regex.match(x).group()))
    idx = pd.notnull(df["Cabin"])
    df.loc[idx, "Cabin"] = df.loc[idx, "Cabin"].apply(lambda x: x[0])
    regex = re.compile(r"\d+")
    df["Ticket_is_digits"] = df["Ticket"].str.match(regex)
    df["Ticket"] = df["Ticket"].apply(lambda x: len(x))
    return df

def encode_categorical(
        df_train: pd.DataFrame,
        ) -> Tuple[pd.DataFrame, sklpre.OrdinalEncoder]:
    cat_map = sklpre.OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1)
    df_train[cat_feats] = (cat_map
                           .fit_transform(df_train[cat_feats]))
    df_train[cat_feats] = df_train[cat_feats].fillna(-1).astype(int)
    return df_train, cat_map

def scale_continuous(df_train: pd.DataFrame,
                     ) -> Tuple[pd.DataFrame, sklpre.StandardScaler]:
    std_scl = sklpre.StandardScaler()
    df_train[cont_feats] = std_scl.fit_transform(df_train[cont_feats])
    df_train[cont_feats] = df_train[cont_feats].fillna(0)
    return df_train, std_scl

def encode_onehot(df_train: pd.DataFrame,
                  ) -> Tuple[pd.DataFrame, sklpre.OneHotEncoder]:
    one_enc = sklpre.OneHotEncoder(sparse=False,
                                   handle_unknown="ignore")
    df_aux = pd.DataFrame(
        one_enc.fit_transform(df_train[cat_feats]),
        index=df_train.index)
    df_train.drop(cat_feats, axis=1, inplace=True)
    df_train = df_train.join(df_aux)
    return df_train, one_enc

def discretize(df, age_bins=10, fare_bins=6):
    """
    """
    df[age] = pd.cut(df_X[age], bins=age_bins,
                     labels=np.arange(0, age_bins))
    df[fare] = pd.cut(df_X[fare], bins=fare_bins,
                     labels=np.arange(0, fare_bins))
    return 0

def get_classifier():
    clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            criterion="gini",
            min_samples_split=3,
            min_samples_leaf=5,)
    return clf

def transform(df: pd.DataFrame,
              cat_map: sklpre.OrdinalEncoder,
              std_scl: sklpre.StandardScaler,
              one_enc: sklpre.OneHotEncoder) -> pd.DataFrame:
    df = apply_feature_transform(df)
    df[cat_feats] = cat_map.transform(df[cat_feats])
    df[cat_feats] = df[cat_feats].fillna(-1).astype(int)
    df[cont_feats] = std_scl.transform(df[cont_feats])
    df[cont_feats] = df[cont_feats].fillna(0)
    df_aux = pd.DataFrame(
            one_enc.transform(df[cat_feats]),
            index=df.index)
    df.drop(cat_feats, axis=1, inplace=True)
    df = df.join(df_aux)
    return df

def main(outfn=default_results_fn):
    idx_train, df_train, y_train, id_test, df_test = load_data()
    df_train = apply_feature_transform(df_train)
    df_train, cat_map = encode_categorical(df_train)
    df_train, std_scl = scale_continuous(df_train)
    df_train, one_enc = encode_onehot(df_train)
    clf = get_classifier()
    clf.fit(df_train, y_train.values.ravel())
    df_test = transform(df_test, cat_map, std_scl, one_enc)
    pred = clf.predict(df_test)
    results = pd.DataFrame({
        "PassengerId": id_test.values,
        "Survived": pred.ravel()})
    results.to_csv(outfn, index=False)
    return df_train, y_train, clf, results

