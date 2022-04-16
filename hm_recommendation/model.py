import sys
import os
from typing import List, Dict
import tensorflow as tf

class RankModel(tf.keras.Model):
    def __init__(self, vocabulary: Dict[str, List[str]], **kwargs):
        super().__init__(
               **kwargs)
        self.club_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["club_member_status"],
                name="club_vectorizer")
        self.club_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["club_member_status"]) + 2,
                output_mode="one_hot",
                name="club_encoder")
        self.news_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["fashion_news_frequency"],
                name="news_frequency_vectorizer")
        self.news_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["fashion_news_frequency"]) + 2,
                output_mode="one_hot",
                name="news_frequency_encoder")
        self.type_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["product_type_name"],
                name="product_type_vectorizer")
        self.type_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["product_type_name"]) + 2,
                output_mode="multi_hot",
                name="product_type_encoder")
        self.group_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["product_group_name"],
                name="product_group_vectorizer")
        self.group_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["product_group_name"]) + 2,
                output_mode="multi_hot",
                name="product_group_encoder")
        self.graphical_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["graphical_appearance_name"],
                name="graphical_vectorizer")
        self.graphical_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["graphical_appearance_name"]) + 2,
                output_mode="multi_hot",
                name="graphical_encoder")
        self.colour_group_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["colour_group_name"],
                name="colour_group_vectorizer")
        self.colour_group_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["colour_group_name"]) + 2,
                output_mode="multi_hot",
                name="colour_group_encoder")
        self.colour_value_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["perceived_colour_value_name"],
                name="colour_value_vectorizer")
        self.colour_value_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["perceived_colour_value_name"]) + 2,
                output_mode="multi_hot",
                name="colour_value_encoder")
        self.colour_master_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["perceived_colour_master_name"]) + 2,
                output_mode="multi_hot",
                name="colour_master_encoder")
        self.department_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["department_name"],
                name="department_vectorizer")
        self.department_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["department_name"]) + 2,
                output_mode="multi_hot",
                name="department_encoder")
        self.index_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["index_name"],
                name="index_vectorizer")
        self.index_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["index_name"]) + 2,
                output_mode="multi_hot",
                name="index_encoder")
        self.index_group_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["index_group_name"],
                name="index_group_vectorizer")
        self.index_group_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["index_group_name"]) + 2,
                output_mode="multi_hot",
                name="index_group_encoder")
        self.section_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["section_name"],
                name="secion_vectorizer")
        self.section_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["section_name"]) + 2,
                output_mode="multi_hot",
                name="section_encoder")
        self.garment_vec = tf.keras.layers.TextVectorization(
                standardize=None,
                split=None,
                vocabulary=vocabulary["garment_group_name"],
                name="garment_vectorizer")
        self.garment_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["garment_group_name"]) + 2,
                output_mode="multi_hot",
                name="garment_encoder")
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.dense0 = tf.keras.layers.Dense(
                units=len(vocabulary["article_id"]) // 1000,
                activation="sigmoid",
                use_bias=True,
                name="dense0")
        self.dense1 = tf.keras.layers.Dense(
                units=len(vocabulary["article_id"]),
                activation="sigmoid",
                use_bias=True,
                name="dense1")
        self.softmax = tf.keras.layers.Softmax(name="last")
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.compile(
                loss={"target": loss,
                      "article_id": None},
                optimizer=opt,
                #metrics={
                )

    def call(self, inputs):
        #xchannel = inputs["sales_channel_id"]
        xprice = inputs["price"]
        xclub = inputs["club_member_status"]
        xclub = self.club_vec(xclub)
        xclub = self.club_encoder(xclub)
        xnews = inputs["fashion_news_frequency"]
        xnews = self.news_vec(xnews)
        xnews = self.news_encoder(xnews)
        xtype = inputs["product_type_name"]
        xtype = self.type_vec(xtype)
        xtype = self.type_encoder(xtype)
        xgroup = inputs["product_group_name"]
        xgroup = self.group_vec(xgroup)
        xgroup = self.group_encoder(xgroup)
        xgraphical = inputs["graphical_appearance_name"]
        xgraphical = self.graphical_vec(xgraphical)
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourgroup = inputs["colour_group_name"]
        xcolourgroup = self.colour_group_vec(xcolourgroup)
        xcolourgroup = self.colour_group_encoder(xcolourgroup)
        xcolourvalue = inputs["perceived_colour_value_name"]
        xcolourvalue = self.colour_group_vec(xcolourvalue)
        xcolourvalue = self.colour_group_encoder(xcolourvalue)
        xcolourmaster = inputs["perceived_colour_master_name"]
        xcolourmaster = self.colour_master_vec(xcolourmaster)
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        xdepartment = inputs["department_name"]
        xdepartment = self.department_vec(xdepartment)
        xdepartment = self.department_encoder(xdepartment)
        xindex = inputs["index_name"]
        xindex = self.index_vec(xindex)
        xindex = self.index_encoder(xindex)
        xindexgroup = inputs["index_group_name"]
        xindexgroup = self.index_group_vec(xindexgroup)
        xindexgroup = self.index_group_encoder(xindexgroup)
        xsection = inputs["section_name"]
        xsection = self.section_vec(xsection)
        xsection = self.section_encoder(xsection)
        xgarment = inputs["garment_group_name"]
        xgarment = self.garment_vec(xgarment)
        xgarment = self.garment_encoder(xgarment)
        #xdetail = inputs["detail_desc"]
        x = self.cat([
                xprice,
                xclub,
                xnews,
                xtype,
                xgroup,
                xgraphical,
                xcolourvalue,
                xcolourmaster,
                xdepartment,
                xindex,
                xindexgroup,
                xsection,
                xgarment,])
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.softmax(x)
        return {"target": x, "article_id": None}

