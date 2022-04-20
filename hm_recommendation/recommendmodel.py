"""
"""
import sys
import os
from typing import List, Dict, Any
import tensorflow as tf
import tensorflow_recommenders as tfrs

class CustomerModel(tf.keras.Model):
    def __init__(self, vocabulary, config, **kwargs):
        super().__init__(**kwargs)
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["customer_id"]),
                config["factor_dim"])

    def call(self, inputs):
        x = inputs["customer_id"]
        x = self.customer_id_lookup(x)
        x = self.emb(x)
        return x

class ArticleModel(tf.keras.Model):
    def __init__(self,
                 vocabulary,
                 lookups,
                 config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        output_type = "one_hot"
        self.lookups = lookups
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["article_id"]) + 1,
                config["article_embedding_dim"])
        self.flatten = tf.keras.layers.Flatten()
        self.group_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["product_group_name"]),
                output_mode=output_type,
                name="product_group_encoder")
        self.graphical_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["graphical_appearance_name"]),
                output_mode=output_type,
                name="graphical_encoder")
        self.colour_master_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["perceived_colour_master_name"]),
                output_mode=output_type,
                name="colour_master_encoder")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.dense0 = tf.keras.layers.Dense(
                units=config["factor_dim"],
                activation="linear",
                use_bias=False,
                name="dense0")

    def call(self, inputs):
        x = inputs["article_id"]
        x = self.emb(x)
        x = self.flatten(x)
        xgroup = self.lookups["product_group_name"].lookup(
                inputs["article_id"])
        xgroup = self.group_encoder(xgroup)
        xgraphical = self.lookups["graphical_appearance_name"].lookup(
                inputs["article_id"])
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourmaster = self.lookups["perceived_colour_master_name"].lookup(
                inputs["article_id"])
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster])
        x = self.batch_norm(x)
        x = self.dense0(x)
        return x

class SequentialQueryModel(tf.keras.Model):
    def __init__(self,
                 vocabulary,
                 lookups,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        self.lookups = lookups
        self.group_encoder = tf.keras.layers.Embedding(
                len(vocabulary["product_group_name"]),
                len(vocabulary["product_group_name"]) // 2,
                name="product_group_encoder")
        self.group_norm = tf.keras.layers.BatchNormalization()
        self.graphical_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["graphical_appearance_name"],
                name="graphical_vectorizer")
        self.graphical_encoder = tf.keras.layers.Embedding(
                len(vocabulary["graphical_appearance_name"]),
                len(vocabulary["graphical_appearance_name"]) // 2,
                name="graphical_encoder")
        self.graphical_norm = tf.keras.layers.BatchNormalization()
        self.colour_master_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.Embedding(
                len(vocabulary["perceived_colour_master_name"]),
                len(vocabulary["perceived_colour_master_name"]) // 2,
                name="colour_master_encoder")
        self.colour_master_norm = tf.keras.layers.BatchNormalization()
        self.club_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["club_member_status"]),
                output_mode="one_hot",
                name="club_encoder")
        self.news_encoder = tf.keras.layers.CategoryEncoding(
                num_tokens=len(vocabulary["fashion_news_frequency"]),
                output_mode="one_hot",
                name="news_frequency_encoder")
        self.batch_norm0 = tf.keras.layers.BatchNormalization()
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["article_id"]) + 1,
                config["article_embedding_dim"],
                name="article_embedding")
        self.gru = tf.keras.layers.GRU(config["gru_dim"])
        self.dense0 = tf.keras.layers.Dense(
                units=config["factor_dim"],
                activation="linear",
                use_bias=False,
                name="dense0")

    def call(self, inputs):
        x = inputs["article_id_hist"]
        x = self.emb(x)
        xgroup = self.lookups["product_group_name"].lookup(
                inputs["article_id_hist"])
        xgroup = self.group_encoder(xgroup)
        xgroup = self.group_norm(xgroup)
        xgraphical = self.lookups["graphical_appearance_name"].lookup(
                inputs["article_id_hist"])
        xgraphical = self.graphical_encoder(xgraphical)
        xgraphical = self.graphical_norm(xgraphical)
        xcolourmaster = self.lookups["perceived_colour_master_name"].lookup(
                inputs["article_id_hist"])
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        xcolourmaster = self.colour_master_norm(xcolourmaster)
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster])
        x = self.batch_norm0(x)
        x = self.gru(x)
        xclub = self.lookups["club_member_status"].lookup(
                inputs["customer_id"])
        xclub = self.club_encoder(xclub)
        xnews = self.lookups["fashion_news_frequency"].lookup(
                inputs["customer_id"])
        xnews = self.news_encoder(xnews)
        x = self.cat([x, xclub, xnews])
        x = self.batch_norm1(x)
        x = self.dense0(x)
        return x

class RetrievalModel(tfrs.Model):
    def __init__(self,
                 vocabulary,
                 articles_ds,
                 lookups,
                 config, **kwargs):
        super().__init__(**kwargs)
        self.customer_model = SequentialQueryModel(
                vocabulary,
                lookups,
                config)
        self.article_model = ArticleModel(
                vocabulary,
                lookups,
                config)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                articles_ds
                    .batch(config["top_k_batch_size"])
                    .map(self.article_model)))
        opt = tf.keras.optimizers.Adagrad(learning_rate=0.1)
        self.compile(optimizer=opt)

    def compute_loss(self, features, training=False) -> tf.Tensor:
        return self.task(
            self.customer_model(features),
            self.article_model(features))

