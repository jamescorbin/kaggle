"""
kaggle competitions download -c tpu-getting-started
"""
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

def get_model_preprocessing_layers(
            strategy: "strategy",
            imagey: int,
            imagex: int,
            classes: int,
            seed: int=50,
            ) -> tf.keras.Model:
    """
    Default version of TensorFlow
    on Kaggle does not have
    image preprocessing layers.
    """
    with strategy.scope():
        tf.random.set_seed(seed)
        one = tf.constant([1], dtype=tf.int32)
        dimx = tf.constant(imagex, dtype=tf.int32)
        dimy = tf.constant(imagey, dtype=tf.int32)
        dx_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        dy_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        rot_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        shear_y_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.10,
                    dtype=tf.float32))
        shear_x_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.10,
                    dtype=tf.float32))
        zoom_0 = tf.math.abs(tf.random.normal(one,
                    mean=0.0,
                    stddev=0.15,
                    dtype=tf.float32))
        random_translation_layer = tf.keras.layers.RandomTranslation(
                    dx_0,
                    dy_0,
                    name="random_translation")
        random_flip_layer = tf.keras.layers.RandomFlip(
                    name="random_flip")
        random_rotation_layer = tf.keras.layers.RandomRotation(
                    rot_0,
                    name="random_rotation")
        random_shear_height = tf.keras.layers.RandomHeight(
                    factor=shear_y_0,
                    name="random_shear_height")
        random_shear_width = tf.keras.layers.RandomWidth(
                    factor=shear_x_0,
                    name="random_shear_width")
        random_zoom_layer = tf.keras.layers.RandomZoom(
                    height_factor=zoom_0,
                    width_factor=zoom_0,
                    name="random_zoom")
        center_crop_layer = tf.keras.layers.CenterCrop(
                    height=imagey,
                    width=imagex,
                    name="center_crop")
        rnet = DenseNet201(
                    input_shape=(imagey, imagex, 3),
                    weights="imagenet",
                    include_top=False,)
        input1 = tf.keras.Input(
                    shape=(imagey, imagex, 3),
                    dtype=tf.float32)
        inputs = {"image": input1}
        input2 = tf.keras.Input(shape=(), dtype=tf.string)
        inputs["id"] = input2
        pooling = tf.keras.layers.GlobalAveragePooling2D(
                name="pooling")
        out_layer = tf.keras.layers.Dense(
                classes,
                activation="softmax",
                dtype=tf.float32)
        x = input1
        x = random_translation_layer(x)
        x = random_flip_layer(x)
        x = random_rotation_layer(x)
        #x = random_shear_height(x)
        #x = center_crop_layer(x)
        #x = random_shear_width(x)
        #x = center_crop_layer(x)
        x = random_zoom_layer(x)
        x = center_crop_layer(x)
        x = rnet(x)
        x = pooling(x)
        x = out_layer(x)
        label = tf.reshape(tf.math.top_k(x, k=1).indices, shape=[-1])
        outputs = {"class": x, "label": label}
        outputs["id"] = input2
        model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs,
                name="flower_model")
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        opt = tf.keras.optimizers.Adam()
    model.compile(
            optimizer=opt,
            loss={"class": loss},
            metrics={"class": [metric]})
    return model

