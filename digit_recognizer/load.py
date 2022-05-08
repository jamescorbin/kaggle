import sys
import os
import tensorflow as tf

def get_training_dataset(x_train, y_train, config):
    x_ds = tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(x_train),
        tf.data.Dataset.from_tensor_slices(y_train)))
    x_train = (x_ds
            .enumerate()
            .filter(lambda x, y:
                    x % config["split_mod_k"] <= config["train_k"])
            .map(lambda x, y: y)
            .shuffle(config["shuffle_buffer"]))
    x_valid = (x_ds
            .enumerate()
            .filter(lambda x, y:
                    (x % config["split_mod_k"] > config["train_k"])
                    & (x % config["split_mod_k"] <= config["valid_k"]))
            .map(lambda x, y: y))
    x_test = (x_ds
            .enumerate()
            .filter(lambda x, y:
                    x % config["split_mod_k"] > config["valid_k"])
            .map(lambda x, y: y))
    return x_train, x_valid, x_test
