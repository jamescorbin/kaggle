import sys
import os
import tensorflow as tf
import transform

def get_dataset(x_ds, config):
    _f1 = lambda x: {"image": x["image"], "id": x["id"]}
    _f2 = lambda x: {"class": x["class"]}
    x_ds = tf.data.Dataset.zip((x_ds.map(_f1), x_ds.map(_f2)))
    _g1 = lambda x, y: x % config["split_mod_k"] <= config["train_k"]
    _g2 = (lambda x, y:
            (x % config["split_mod_k"] > config["train_k"])
            & (x % config["split_mod_k"] <= config["valid_k"]))
    _g3 = lambda x, y: x % config["split_mod_k"] > config["valid_k"]
    _h = lambda x, y: y
    x_train = (x_ds
            .enumerate()
            .filter(_g1)
            .map(_h))
    x_valid = (x_ds
            .enumerate()
            .filter(_g2)
            .map(_h))
    x_test = (x_ds
            .enumerate()
            .filter(_g3)
            .map(_h))
    x_train = transform.augment_train(x_train, config)
    return x_train, x_valid, x_test


