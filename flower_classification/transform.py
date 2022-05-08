import sys
import os
import tensorflow as tf
import imageaug

@tf.function
def transform(x: tf.Tensor) -> tf.Tensor:
    return imageaug.apply_affine_transform(
            x,
            imageaug.get_random_transformation())

def augment_train(x_train: tf.data.Dataset,
                  config) -> tf.data.Dataset:
    x_train = x_train.repeat(config["repeat"])
    x_train = x_train.shuffle(config["shuffle_buffer"])
    x_train = x_train.batch(
            config["batch_dim"],
            drop_remainder=True)
    _f = (lambda x, y:
                  ({"image": transform(x["image"]),
                      "id": x["id"]},
                  y))
    x_train.map(_f)
    return x_train

def test_image_transform(ds, config) -> None:
    import matplotlib.pyplot as plt
    import tensorflow_datasets as tfds
    dim = config["image_shape"][0]
    #ds = ds.map(lambda x, y: x["image"])
    ds = ds.map(lambda x: x["image"])
    ims = []
    ds = ds.take(1).repeat(9).batch(1)
    for k in ds.unbatch().take(1):
        ims.append(k)
    ds = ds.map(transform)
    ds = ds.unbatch().take(3)
    for k in ds:
        ims.append(k)
    fix, ax = plt.subplots(2, 2)
    ax[0][0].imshow(ims[0])
    ax[1][0].imshow(ims[1])
    ax[0][1].imshow(ims[2])
    ax[1][1].imshow(ims[3])
    plt.savefig("test_image.png")

