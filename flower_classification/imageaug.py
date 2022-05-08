import sys
import os
import math
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import tensorflow_datasets as tfds

DEG_TO_RAD = 2.0 * math.pi / 360.0

@tf.function
def get_random_rotation(th_std: tf.Tensor) -> tf.Tensor:
    rot = tf.random.normal(
            [1],
            mean=0.0,
            stddev=th_std * DEG_TO_RAD,
            dtype=tf.float32)
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    sn = tf.math.sin(rot)
    cn = tf.math.cos(rot)
    rot_matrix = tf.reshape(
            [cn, sn, zero,
            -1.0 * sn, cn, zero,
            zero, zero, one],
            (3, 3))
    return rot_matrix

@tf.function
def get_random_shear_height(yshr_std: tf.Tensor) -> tf.Tensor:
    shr = tf.random.normal(
            [1],
            mean=0.0,
            stddev=yshr_std,
            dtype=tf.float32)
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    shear_matrix = tf.reshape(
            [one, shr, zero,
            zero, one, zero,
            zero, zero, one],
            (3, 3))
    return shear_matrix

@tf.function
def get_random_shear_width(xshr_std: tf.Tensor) -> tf.Tensor:
    shr = tf.random.normal(
            [1],
            mean=0.0,
            stddev=xshr_std,
            dtype=tf.float32)
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    shear_matrix = tf.reshape(
            [one, zero, zero,
            shr, one, zero,
            zero, zero, one],
            (3, 3))
    return shear_matrix

@tf.function
def get_random_shift(yd_std: tf.Tensor, xd_std: tf.Tensor) -> tf.Tensor:
    h_shift = tf.random.normal(
            [1],
            mean=0.0,
            stddev=xd_std,
            dtype=tf.float32)
    w_shift = tf.random.normal(
            [1],
            mean=0.0,
            stddev=yd_std,
            dtype=tf.float32)
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    shift_matrix = tf.reshape(
            [one, zero, h_shift,
            zero, one, w_shift,
            zero, zero, one],
            (3, 3))
    return shift_matrix

@tf.function
def get_random_scale(xscl_std: tf.Tensor, yscl_std: tf.Tensor) -> tf.Tensor:
    one = tf.constant([1.0], dtype=tf.float32)
    zero = tf.constant([0.0], dtype=tf.float32)
    h_zoom = tf.random.normal(
            [1],
            mean=1.0,
            stddev=xscl_std,
            dtype=tf.float32)
    w_zoom = tf.random.normal(
            [1],
            mean=1.0,
            stddev=yscl_std,
            dtype=tf.float32)
    zoom_matrix = tf.reshape(
            [1.0 / h_zoom, zero, zero,
            zero, 1.0 / w_zoom, zero,
            zero, zero, one],
            (3, 3))
    return zoom_matrix

@tf.function
def get_random_transformation(
            th_std: tf.Tensor=tf.constant(15.0),
            yshr_std: tf.Tensor=tf.constant(0.15),
            xshr_std: tf.Tensor=tf.constant(0.15),
            xd_std: tf.Tensor=tf.constant(20.0),
            yd_std: tf.Tensor=tf.constant(20.0),
            sclx0: tf.Tensor=tf.constant(0.15),
            scly0: tf.Tensor=tf.constant(0.15),
            ) -> tf.Tensor:
    rot_matrix = get_random_rotation(th_std)
    height_shear_matrix = get_random_shear_height(yshr_std)
    width_shear_matrix = get_random_shear_width(xshr_std)
    shift_matrix = get_random_shift(xd_std=xd_std,
                                    yd_std=yd_std)
    zoom_matrix = get_random_scale(sclx0, scly0)
    return tf.linalg.matmul(rot_matrix,
            tf.linalg.matmul(height_shear_matrix,
                tf.linalg.matmul(width_shear_matrix,
                    tf.linalg.matmul(zoom_matrix,
                        shift_matrix))))

@tf.function
def apply_affine_transform(
        img: tf.Tensor,
        mat: tf.Tensor,
        ) -> tf.Tensor:
    """
    This method was used due to the
    tf.keras.preprocessing.image.apply_affine_transformation
    being a numpy function making that not amenable
    to launching on a TPU.
    There are methods like tf.keras.layers.RandomRotation
    that solve this issue.

    Modified to accept batched inputs.
    If not batched, apply batch_size == 1
    """
    zero = tf.constant(0, dtype=tf.int32)
    one = tf.constant(1, dtype=tf.int32)
    dim = tf.gather(tf.shape(img), one)
    d = tf.cast(dim // 2, tf.float32)
    iidxs = (tf.tile(
        tf.reshape(
            tf.range(dim, dtype=tf.float32),
                (dim, 1)), [1, dim]) - d)
    jidxs = (tf.tile(
        tf.reshape(
            tf.range(dim, dtype=tf.float32),
                (1, dim)), [dim, 1]) - d)
    affidx = tf.ones(shape=(dim, dim))
    idxs = tf.stack([iidxs, jidxs, affidx], axis=2)
    k1 = tf.tensordot(idxs, mat, axes=[[-1], [-1]])
    k2 = tf.cast(k1 + d, dtype=tf.int32)
    k3 = tf.reshape(k2, (dim * dim, 3))
    k4 = tf.clip_by_value(k3, 0, dim - 1)
    k5 = tf.slice(k4, (0, 0), (dim * dim, 2))
    # k5 removes affine shift dummy dimension
    k6 = tf.tile(
        tf.reshape(k5, (1, dim * dim, 2)),
        [tf.gather(tf.shape(img), zero), 1, 1])
    # tile for batch size
    gat = tf.gather_nd(
            params=img,
            indices=k6,
            batch_dims=1,
            )
    new_img = tf.reshape(gat, (-1, dim, dim, 3),)
    return new_img
