import sys
import os
import mlflow
import tensorflow as tf
import load
import tweetmodel

def run_training_loop(ds, config):
    config_fn = "config.json"
    tfrec_dir = "data/tfrec"
    tfboard_log_dir = "data/tfboard"
    model_save_pt = "data/model"
    model_checkpoint_pt = "data/checkpoint"
    strategy = tf.distribute.get_strategy()
    mlflow.set_tracking_uri("file:///home/jec/Desktop/artifacts")
    mlflow.set_experiment("disaster_tweets")
    mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_model_signatures=True)
    batch_size = config["batch_dim"] * strategy.num_replicas_in_sync
    with mlflow.start_run():
        with strategy.scope():
            xtrain, xvalid, xtest = load.get_tfds(
                    ds,
                    config=config,)
            model = tweetmodel.TweetModel(config)
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
            hist_0 = model.fit(
                    xtrain
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(tf.data.AUTOTUNE),
                    validation_data=xvalid
                        .batch(2**10)
                        .prefetch(tf.data.AUTOTUNE),
                    epochs=config["epochs_init"],
                    callbacks=callbacks)
            model.bert_layer.trainable = True
            optimizer = tf.keras.optimizers.Adam(
                    learning_rate=1e-5)
            metrics = [tf.keras.metrics.BinaryAccuracy()]
            loss = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False)
            model.compile(
                    optimizer=optimizer,
                    loss={"target": loss},
                    metrics={"target": metrics},)
            hist_1 = model.fit(
                    xtrain
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(tf.data.AUTOTUNE),
                    validation_data=xvalid
                        .batch(2**10)
                        .prefetch(tf.data.AUTOTUNE),
                    epochs=config["epochs_tune"],
                    callbacks=callbacks)
            test_eval = model.evaluate(
                    xtest.batch(2**10),
                    callbacks=callbacks,
                    return_dict=True)
        for key, value in config.items():
            mlflow.log_param(key, value)
        for key, value in test_eval.items():
            mlflow.log_metric(f"test_{key}", value)
        model.save(model_save_pt)
        mlflow.log_artifact(config_fn)


