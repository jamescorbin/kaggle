from typing import Dict
import tensorflow as tf
import numpy as np

def _byteslist(value):
    return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=value))
def _int64list(value):
    return tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
def _floatlist(value):
    return tf.train.Feature(
            float_list=tf.train.FloatList(value=value))

def serialize_example(ds: Dict[str, np.array]):
    feature = {
        "t_dat": _byteslist(ds["t_dat"]),
        "customer_id": _byteslist(ds["customer_id"]),
        "fn": _floatlist(ds["FN"]),
        "active": _floatlist(ds["Active"]),
        "club_member_status": _byteslist(ds["club_member_status"]),
        "fashion_news_frequency": _byteslist(ds["fashion_news_frequency"]),
        "age": _floatlist(ds["age"]),
        "price": _floatlist(ds["price"]),
        "sales_channel_id": _int64list(ds["sales_channel_id"]),
        "article_id": _byteslist(ds["article_id"]),
        "product_type_name": _byteslist(ds["product_type_name"]),
        "product_group_name": _byteslist(ds["product_group_name"]),
        "graphical_appearance_name":
                _byteslist(ds["graphical_appearance_name"]),
        "colour_group_name":
                _byteslist(ds["colour_group_name"]),
        "perceived_colour_value_name":
                _byteslist(ds["perceived_colour_value_name"]),
        "perceived_colour_master_name":
                _byteslist(ds["perceived_colour_master_name"]),
        "department_name":
                _byteslist(ds["department_name"]),
        "index_name":
                _byteslist(ds["index_name"]),
        "index_group_name":
                _byteslist(ds["index_group_name"]),
        "section_name":
                _byteslist(ds["section_name"]),
        "garment_group_name":
                _byteslist(ds["garment_group_name"]),
        "detail_desc":
                _byteslist(ds["detail_desc"]),
        "target": _byteslist(ds["target"]),}
    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(ds):
    tf_string = tf.py_function(
        serialize_example,
        ds,
        tf.string)
    return tf.reshape(tf_string, ())

def parse(example):
    ts_len = 5
    feature_description = {
        "t_dat": tf.io.FixedLenFeature([1], tf.string),
        "customer_id": tf.io.FixedLenFeature([1], tf.string),
        "fn": tf.io.FixedLenFeature([1], tf.float32),
        "active": tf.io.FixedLenFeature([1], tf.float32),
        "club_member_status": tf.io.FixedLenFeature([1], tf.string),
        "fashion_news_frequency": tf.io.FixedLenFeature([1], tf.string),
        "age": tf.io.FixedLenFeature([1], tf.float32),
        "price": tf.io.FixedLenFeature([ts_len], tf.float32),
        "sales_channel_id": tf.io.FixedLenFeature([ts_len], tf.int64),
        "article_id": tf.io.FixedLenFeature([ts_len], tf.string),
        "product_type_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "product_group_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "graphical_appearance_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "colour_group_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "perceived_colour_value_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "perceived_colour_master_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "department_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "index_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "index_group_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "section_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "garment_group_name": tf.io.FixedLenFeature([ts_len], tf.string),
        "detail_desc": tf.io.FixedLenFeature([ts_len], tf.string),
        "target": tf.io.FixedLenFeature([1], tf.string)}
    return tf.io.parse_single_example(example, feature_description)

def serialize():
    with tf.io.TFRecordWriter(out_fp) as writer:
        writer.write(tf_serialize_example(example))



