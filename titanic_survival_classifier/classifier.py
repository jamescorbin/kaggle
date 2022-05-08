import sys
import os
import sklearn.ensemble

def get_classifier():
    clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            criterion="gini",
            min_samples_split=3,
            min_samples_leaf=5,)
    return clf
