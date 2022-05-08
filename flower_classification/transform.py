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

def test_image_transform() -> None:
    import matplotlib.pyplot as plt
    dims = [192, 224, 331, 512]
    dim = dims[-1]
    ds = get_dataset(dim,
                    repeat=None,
                    buffer_size=None,
                    batch_size=None,
                    apply_transformation=False,
                    train_test="train",)
    ds = ds.map(lambda x, y: x)
    ds = ds.map(lambda x: x["image"])
    ds = ds.skip(100).take(1).repeat(2)
    ds = tfds.as_numpy(ds)
    it = iter(ds)
    ds0 = next(it)
    ds1 = transform(next(it))
    fix, ax = plt.subplots(1, 2)
    ax[0].imshow(ds0)
    ax[1].imshow(ds1)
    plt.savefig("test_image.png")


