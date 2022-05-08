import sys
import os
import tensorflow as tf
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import load
import digitmodel

def run_training_loop(x_train, y_train, config):
    strategy = tf.distribute.get_strategy()
    batch_size = config["batch_dim"] * strategy.num_replicas_in_sync
    tfboard_log_dir = "./data/tfboard"
    model_checkpoint_pt = "./data/checkpoint"
    with strategy.scope():
        x_train, x_valid, x_test = (
            load.get_training_dataset(x_train, y_train, config))
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
                save_weights_only=False,
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
        model = digitmodel.build_model(config)
        hist = model.fit(
                x_train
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE),
                validation_data=x_valid.batch(batch_size),
                epochs=config["epochs"],
                callbacks=callbacks)
        eval_test = model.evaluate(
                x_test.batch(batch_size),
                callbacks=callbacks,
                return_dict=True)
    return model, hist, eval_test
