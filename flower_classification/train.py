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
    config_fn = "config.json"
    tfrec_dir = "data/tfrec"
    tfboard_log_dir = "data/tfboard"
    model_save_pt = "data/model"
    model_checkpoint_pt = "data/checkpoint"
    tf.random.set_seed(config["seed"])
    tpu, strategy = get_strategy()
    mlflow.set_tracking_uri("file:///home/jec/Desktop/artifacts")
    mlflow.set_experiment("disaster_tweets")
    mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_model_signatures=True)
    batch_size = config["batch_size"] * strategy.num_replicas_in_sync
    with strategy.scope():
        with mlflow.start_run():
            x_train, x_valid, x_test = get_training_dataset(config)
            x_train = augment_train(x_train, batch_size, config=config)
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
            model = FlowerModel(
                    config=config,
                    name="flower_model")
            with mlflow.start_run(nested=True):
                hist0 = model.fit(
                        ds_train,
                        validation_data=x_valid.batch(batch_size),
                        callbacks=callbacks,
                        epochs=config["epochs_init"])
            with mlflow.start_run(nested=True):
                model.set_trainable_recompile()
                hist1 = model.fit(
                        ds_train,
                        validation_data=ds_valid.batch(batch_size),
                        epochs=config["epochs_tune"],
                        callbacks=[early_stopping])
                test_eval = model.evaluate(
                        x_test.batch(batch_size),
                        callbacks=callbacks)
                for key, value in config.items():
                    mlflow.log_param(key, value)
                for key, value in test_eval.items():
                    mlflow.log_metric(f"test_{key}", value)
                model.save(model_save_pt)
                mlflow.log_artifact(config_fn)
