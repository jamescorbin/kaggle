import sys
import os
import json
from typing import Tuple, Optional
import tensorflow as tf
import mlflow
import extract
import load
import flowerclassifier

def get_strategy() -> Tuple[Optional["tpu"], Optional["strategy"]]:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    return tpu, strategy

def run_training_loop():
    config_fn = "config-model.json"
    with open(config_fn, "r") as f:
        config = json.load(f)
    tfrec_dir = "data/tfrec"
    tfboard_log_dir = "data/tfboard"
    model_save_pt = "data/model"
    model_checkpoint_pt = "data/checkpoint"
    ds = extract.get_training_dataset(config)
    tf.random.set_seed(config["seed"])
    tpu, strategy = get_strategy()
    mlflow.set_tracking_uri("file:///home/jec/Desktop/artifacts")
    mlflow.set_experiment("flower_classification")
    mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_model_signatures=True)
    batch_size = config["batch_dim"] * strategy.num_replicas_in_sync
    with strategy.scope():
        with mlflow.start_run():
            x_train, x_valid, x_test = load.get_dataset(ds, config)
            tfboard = tf.keras.callbacks.TensorBoard(
                    log_dir=tfboard_log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    write_steps_per_second=False,
                    update_freq="epoch",
                    profile_batch=0,
                    embeddings_freq=1,
                    embeddings_metadata=None,)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    model_checkpoint_pt,
                    monitor="val_loss",
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=True,
                    mode="auto",
                    save_freq="epoch",
                    options=None,
                    initial_value_threshold=None,)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=2,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True)
            callbacks = [tfboard, model_checkpoint, early_stopping]
            model = flowerclassifier.FlowerModel(
                    config=config,
                    name="flower_model")
            with mlflow.start_run(nested=True):
                hist0 = model.fit(
                        x_train,
                        validation_data=x_valid.batch(2**10),
                        callbacks=callbacks,
                        epochs=config["epochs_init"])
            with mlflow.start_run(nested=True):
                model.set_trainable_recompile()
                hist1 = model.fit(
                        x_train,
                        validation_data=x_valid.batch(2**10),
                        epochs=config["epochs_tune"],
                        callbacks=callbacks)
                test_eval = model.evaluate(
                        x_test.batch(batch_size),
                        callbacks=callbacks,
                        return_dict=True)
                for key, value in config.items():
                    mlflow.log_param(key, value)
                for key, value in test_eval.items():
                    mlflow.log_metric(f"test_{key}", value)
                model.save(model_save_pt)
                mlflow.log_artifact(config_fn)
