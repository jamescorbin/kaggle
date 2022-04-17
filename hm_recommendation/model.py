"""
"""
import sys
import os
from typing import List, Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs

class CustomerModel(tf.keras.Model):
    def __init__(self, vocabulary, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.lookup = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["customer_id"],
                mask_token=None)
        #self.club_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["club_member_status"],
        #        name="club_vectorizer")
        #self.club_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["club_member_status"]) + 1,
        #        output_mode="one_hot",
        #        name="club_encoder")
        #self.news_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["fashion_news_frequency"],
        #        name="news_frequency_vectorizer")
        #self.news_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["fashion_news_frequency"]) + 2,
        #        output_mode="one_hot",
        #        name="news_frequency_encoder")
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["customer_id"]) + 1,
                embedding_dim)

    def call(self, inputs):
        x = inputs["customer_id"]
        x = self.lookup(x)
        x = self.emb(x)
        return x

class ArticleModel(tf.keras.Model):
    def __init__(self, vocabulary, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        output_type = "one_hot"
        self.lookup = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["article_id"],
                mask_token=None)
        self.group_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["product_group_name"],
                name="product_group_vectorizer")
        self.group_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["product_group_name"]),
                output_mode=output_type,
                name="product_group_encoder")
        self.graphical_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["graphical_appearance_name"],
                name="graphical_vectorizer")
        self.graphical_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["graphical_appearance_name"]),
                output_mode=output_type,
                name="graphical_encoder")
        self.colour_master_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["perceived_colour_master_name"]),
                output_mode=output_type,
                name="colour_master_encoder")
        #self.type_lookup = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["product_type_name"],
        #        name="product_type_vectorizer")
        #self.type_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["product_type_name"]) + 1,
        #        output_mode="count",
        #        name="product_type_encoder")
        #self.colour_group_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["colour_group_name"],
        #        name="colour_group_vectorizer")
        #self.colour_group_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["colour_group_name"]) + 1,
        #        output_mode="count",
        #        name="colour_group_encoder")
        #self.colour_value_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["perceived_colour_value_name"],
        #        name="colour_value_vectorizer")
        #self.colour_value_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["perceived_colour_value_name"]) + 1,
        #        output_mode="count",
        #        name="colour_value_encoder")
        #self.department_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["department_name"],
        #        name="department_vectorizer")
        #self.department_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["department_name"]) + 1,
        #        output_mode="count",
        #        name="department_encoder")
        #self.index_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["index_name"],
        #        name="index_vectorizer")
        #self.index_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["index_name"]) + 1,
        #        output_mode="count",
        #        name="index_encoder")
        #self.index_group_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["index_group_name"],
        #        name="index_group_vectorizer")
        #self.index_group_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["index_group_name"]) + 1,
        #        output_mode="count",
        #        name="index_group_encoder")
        #self.section_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["section_name"],
        #        name="secion_vectorizer")
        #self.section_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["section_name"]) + 1,
        #        output_mode="count",
        #        name="section_encoder")
        #self.garment_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["garment_group_name"],
        #        name="garment_vectorizer")
        #self.garment_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["garment_group_name"]) + 1,
        #        output_mode="count",
        #        name="garment_encoder")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["article_id"]) + 1,
                embedding_dim)

    def call(self, inputs):
        x = inputs["article_id"]
        x = self.lookup(x)
        x = self.emb(x)
        xgroup = inputs["product_group_name"]
        xgroup = self.group_vec(xgroup)
        xgroup = self.group_encoder(xgroup)
        xgraphical = inputs["graphical_appearance_name"]
        xgraphical = self.graphical_vec(xgraphical)
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourmaster = inputs["perceived_colour_master_name"]
        xcolourmaster = self.colour_master_vec(xcolourmaster)
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster])
        x = self.batch_norm(x)
        return x

class SequentialQueryModel(tf.keras.Model):
    def __init__(self, vocabulary, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        output_type = "count"
        self.lookup = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["article_id"],
                mask_token=None)
        self.group_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["product_group_name"],
                name="product_group_vectorizer")
        self.group_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["product_group_name"]),
                output_mode=output_type,
                name="product_group_encoder")
        self.graphical_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["graphical_appearance_name"],
                name="graphical_vectorizer")
        self.graphical_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["graphical_appearance_name"]),
                output_mode=output_type,
                name="graphical_encoder")
        self.colour_master_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["perceived_colour_master_name"]),
                output_mode=output_type,
                name="colour_master_encoder")
        #self.type_lookup = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["product_type_name"],
        #        name="product_type_vectorizer")
        #self.type_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["product_type_name"]) + 1,
        #        output_mode="count",
        #        name="product_type_encoder")
        #self.colour_group_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["colour_group_name"],
        #        name="colour_group_vectorizer")
        #self.colour_group_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["colour_group_name"]) + 1,
        #        output_mode="count",
        #        name="colour_group_encoder")
        #self.colour_value_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["perceived_colour_value_name"],
        #        name="colour_value_vectorizer")
        #self.colour_value_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["perceived_colour_value_name"]) + 1,
        #        output_mode="count",
        #        name="colour_value_encoder")
        #self.department_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["department_name"],
        #        name="department_vectorizer")
        #self.department_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["department_name"]) + 1,
        #        output_mode="count",
        #        name="department_encoder")
        #self.index_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["index_name"],
        #        name="index_vectorizer")
        #self.index_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["index_name"]) + 1,
        #        output_mode="count",
        #        name="index_encoder")
        #self.index_group_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["index_group_name"],
        #        name="index_group_vectorizer")
        #self.index_group_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["index_group_name"]) + 1,
        #        output_mode="count",
        #        name="index_group_encoder")
        #self.section_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["section_name"],
        #        name="secion_vectorizer")
        #self.section_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["section_name"]) + 1,
        #        output_mode="count",
        #        name="section_encoder")
        #self.garment_vec = tf.keras.layers.StringLookup(
        #        vocabulary=vocabulary["garment_group_name"],
        #        name="garment_vectorizer")
        #self.garment_encoder = tf.keras.layers.CategoryEncoding(
        #        num_tokens=len(vocabulary["garment_group_name"]) + 1,
        #        output_mode="count",
        #        name="garment_encoder")
        self.club_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["club_member_status"],
                name="club_vectorizer")
        self.club_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["club_member_status"]) + 1,
                output_mode="one_hot",
                name="club_encoder")
        self.news_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["fashion_news_frequency"],
                name="news_frequency_vectorizer")
        self.news_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["fashion_news_frequency"]) + 2,
                output_mode="one_hot",
                name="news_frequency_encoder")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["article_id"]) + 1,
                embedding_dim)
        self.gru = tf.keras.layers.GRU(embedding_dim)
        self.dense0 = tf.keras.layers.Dense(
                units=embedding_dim,
                activation="sigmoid",
                use_bias=True,
                name="dense0")

    def call(self, inputs):
        x = inputs["article_id"]
        x = self.lookup(x)
        x = self.emb(x)
        xgroup = inputs["product_group_name"]
        xgroup = self.group_vec(xgroup)
        xgroup = self.group_encoder(xgroup)
        xgroup = self.batch_norm(xgroup)
        xgraphical = inputs["graphical_appearance_name"]
        xgraphical = self.graphical_vec(xgraphical)
        xgraphical = self.graphical_encoder(xgraphical)
        xgraphical = self.batch_norm(xgraphical)
        xcolourmaster = inputs["perceived_colour_master_name"]
        xcolourmaster = self.colour_master_vec(xcolourmaster)
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        xcolourmaster = self.batch_norm(xcolourmaster)
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster])
        x = self.batch_norm(x)
        x = self.gru(x)
        xclub = inputs["club_member_status"]
        xclub = self.club_vec(xclub)
        xclub = self.club_encoder(xclub)
        xnews = inputs["fashion_news_frequency"]
        xnews = self.news_vec(xnews)
        xnews = self.news_encoder(xnews)
        x = self.cat([x, xclub, xnews])
        x = self.batch_norm(x)
        x = self.dense0(x)
        return x

class RetrievalModel(tfrs.Model):
    def __init__(self, vocabulary, articles_ds, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.customer_model = SequentialQueryModel(vocabulary, embedding_dim)
        self.article_model = ArticleModel(vocabulary, embedding_dim)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                articles_ds.batch(128).map(self.article_model)))
        opt = tf.keras.optimizers.Adagrad(learning_rate=0.1)
        self.compile(optimizer=opt)

    def compute_loss(self, features, training=False) -> tf.Tensor:
        return self.task(
            self.customer_model(features),
            self.article_model(features))

