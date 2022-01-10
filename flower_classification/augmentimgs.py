import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np

@tf.function
def transform(x: tf.Tensor) -> tf.Tensor:
    return apply_affine_transform(x, get_random_transformation())

@tf.function
def get_random_rotation(th0: tf.Tensor=tf.constant(15.0)) -> tf.Tensor:
    rot = tf.random.normal(1,
                        mean=0,
                        stddev= th0 * 2.0 * np.pi / 360.0,
                        dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    zero = tf.constant(0.0, dtype=tf.float32)
    sn = tf.math.sin(rot)
    cn = tf.math.cos(rot)
    rot_matrix = tf.reshape(
                    [cn, sn, zero,
                    -1.0 * sn, cn, zero,
                    zero, zero, one],
                    (3, 3))
    return rot_matrix

@tf.function
def get_random_shear(shr0: tf.Tensor=tf.constant(15.0)) -> tf.Tensor:
    shr = tf.random.normal([1],
                        mean=0,
                        stddev= shr0 * 2.0 * np.pi / 360.0,
                        dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    zero = tf.constant(0.0, dtype=tf.float32)
    sn = tf.math.sin(shr)
    cn = tf.math.cos(shr)
    shear_matrix = tf.reshape(
                [cn * sn + 1.0, cn, zero,
                sn, one, zero,
                zero, zero, one],
                (3, 3))
    return shear_matrix

@tf.function
def get_random_shift(
                xd0: tf.Tensor=tf.constant(10.0),
                yd0: tf.Tensor=tf.constant(10.0),
                ) -> tf.Tensor:
    h_shift = xd0 * tf.random.normal(1, mean=1.0, dtype=tf.float32)
    w_shift = yd0 * tf.random.normal(1, mean=1.0, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    zero = tf.constant(0.0, dtype=tf.float32)
    shift_matrix = tf.reshape(
                [one, zero, h_shift,
                zero, one, w_shift,
                zero, zero, one],
                (3, 3))
    return shift_matrix

@tf.function
def get_random_scale(
                sclx0: tf.Tensor=tf.constant(10.0),
                scly0: tf.Tensor=tf.constant(10.0),
                ) -> tf.Tensor:
    one = tf.constant(1.0, dtype=tf.float32)
    zero = tf.constant(0.0, dtype=tf.float32)
    h_zoom = tf.random.normal(1, stddev=sclx0, dtype=tf.float32)
    w_zoom = tf.random.normal(1, stddev=scly0, dtype=tf.float32)
    zoom_matrix = tf.reshape(
                [1.0 / h_zoom, zero, zero,
                zero, 1.0 / w_zoom, zero,
                zero, zero, one],
                (3, 3))
    return zoom_matrix

@tf.function
def get_random_transformation(
                th0: tf.Tensor=tf.constant(15.0),
                shr0: tf.Tensor=tf.constant(15.0),
                xd0: tf.Tensor=tf.constant(10.0),
                yd0: tf.Tensor=tf.constant(10.0),
                sclx0: tf.Tensor=tf.constant(10.0),
                scly0: tf.Tensor=tf.constant(10.0),
                ) -> tf.Tensor:
    rot_matrix = get_random_rotation(th0)
    shear_matrix = get_random_shear(shr0)
    shift_matrix = get_random_shift(xd0, yd0)
    zoom_matrix = get_random_scale(sclx0, scly0)
    return tf.linalg.matmul(rot_matrix,
            tf.linalg.matmul(shear_matrix,
                tf.linalg.matmul(zoom_matrix,
                    shift_matrix)))

@tf.function
def apply_affine_transform(img: tf.Tensor,
                           mat: tf.Tensor,
                           ) -> tf.Tensor:
    """
    This method was used due to the
    tf.keras.preprocessing.image.apply_affine_transformation
    being a numpy function making that not amenable
    to launching on a TPU.
    There are methods like tf.keras.layers.RandomRotation
    that solve this issue.
    """
    zero = tf.constant(0, dtype=tf.int32)
    dim = tf.gather(tf.shape(img), zero)
    d = tf.cast(dim // 2, tf.float32)
    iidxs = (tf.tile(tf.reshape(tf.range(dim, dtype=tf.float32),
                           (1, dim)), [dim, 1]) - d)
    jidxs = (tf.tile(tf.reshape(tf.range(dim, dtype=tf.float32),
                           (dim, 1)), [1, dim]) - d)
    affidx = tf.ones(shape=(dim, dim))
    idxs = tf.stack([iidxs, jidxs, affidx], axis=2)
    k1 = tf.tensordot(idxs, mat, axes=[[-1], [-1]])
    k2 = tf.cast(k1 + d, dtype=tf.int32)
    k3 = tf.reshape(k2, (dim * dim, 3))
    k4 = tf.clip_by_value(k3, 0, dim - 1)
    k5 = tf.slice(k4, (0, 0), (dim * dim, 2))
    gat = tf.gather_nd(params=img, indices=k5)
    new_img = tf.reshape(gat, (dim, dim, 3))
    return new_img
