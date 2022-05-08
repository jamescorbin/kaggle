"""
kaggle competitions download -c titanic
"""
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
import classifier
import extract
import transform

SEED = 0
np.random.seed(seed=SEED)

def main():
    df_train = extract.load_train_data()
    df_test = extract.load_test_data()
    df_train = transform.apply_feature_transform(df_train)
    df_train, cat_map = transform.encode_categorical(df_train)
    df_train, std_scl = transform.scale_continuous(df_train)
    df_train, one_enc = transform.encode_onehot(df_train)
    clf = classifier.get_classifier()
    clf.fit(
            df_train.drop(["Survived", "PassengerId"], axis=1),
            df_train[["Survived"]].values.ravel())
    df_test = transform.transform(df_test, cat_map, std_scl, one_enc)
    pred = clf.predict(df_test.drop(["PassengerId"], axis=1))
    results = pd.DataFrame({
        "PassengerId": df_test[["PassengerId"]].values.ravel(),
        "Survived": pred.ravel()})

if __name__=="__main__":
    main()
