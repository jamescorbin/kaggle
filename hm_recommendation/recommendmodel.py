"""
"""
import sys
import os
from typing import List, Dict, Any
import tensorflow as tf
import tensorflow_recommenders as tfrs

class CustomerModel(tf.keras.Model):
    def __init__(self,
                 vocabulary,
                 lookups,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["customer_id"]),
                config["factor_dim"])
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = inputs["customer_id"]
        x = self.emb(x)
        x = self.flatten(x)
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
        self.group_encoder = tf.keras.layers.Embedding(
                len(vocabulary["product_group_name"]) + 1,
                len(vocabulary["product_group_name"]) // 2,
                name="product_group_encoder")
        self.section_encoder = tf.keras.layers.Embedding(
                len(vocabulary["section_name"]) + 1,
                len(vocabulary["section_name"]) // 4,
                name="section_encoder")
        self.graphical_encoder = tf.keras.layers.Embedding(
                len(vocabulary["graphical_appearance_name"]) + 1,
                len(vocabulary["graphical_appearance_name"]) // 2,
                name="graphical_encoder")
        self.colour_master_encoder = tf.keras.layers.Embedding(
                len(vocabulary["perceived_colour_master_name"]) + 1,
                len(vocabulary["perceived_colour_master_name"]) // 2,
                name="colour_master_encoder")
        self.batch_norm = tf.keras.layers.BatchNormalization(
                name="article_batch_norm")
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.dense0 = tf.keras.layers.Dense(
                units=config["factor_dim"],
                activation="linear",
                use_bias=False,
                name="dense0")

    def call(self, inputs):
        x = inputs["article_id"]
        x = self.emb(x)
        xsection = self.lookups["section_name"].lookup(
                inputs["article_id"])
        xsection = self.section_encoder(xsection)
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
            xcolourmaster,
            xsection,
            ])
        x = self.flatten(x)
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
        self.price_norm = tf.keras.layers.Normalization(
                mean=config["price_mean"],
                variance=config["price_var"],
                name="price_norm")
        self.group_encoder = tf.keras.layers.Embedding(
                len(vocabulary["product_group_name"]) + 1,
                len(vocabulary["product_group_name"]) // 2,
                name="product_group_encoder")
        self.graphical_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["graphical_appearance_name"],
                name="graphical_vectorizer")
        self.graphical_encoder = tf.keras.layers.Embedding(
                len(vocabulary["graphical_appearance_name"]) + 1,
                len(vocabulary["graphical_appearance_name"]) // 2,
                name="graphical_encoder")
        self.graphical_norm = tf.keras.layers.BatchNormalization()
        self.colour_master_vec = tf.keras.layers.StringLookup(
                vocabulary=vocabulary["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.Embedding(
                len(vocabulary["perceived_colour_master_name"]) + 1,
                len(vocabulary["perceived_colour_master_name"]) // 2,
                name="colour_master_encoder")
        self.colour_master_norm = tf.keras.layers.BatchNormalization()
        self.club_encoder = tf.keras.layers.Embedding(
                len(vocabulary["club_member_status"]) + 1,
                len(vocabulary["club_member_status"]),
                name="club_encoder")
        self.news_encoder = tf.keras.layers.Embedding(
                len(vocabulary["fashion_news_frequency"]) + 1,
                len(vocabulary["fashion_news_frequency"]),
                name="news_frequency_encoder")
        self.batch_norm0 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_0")
        self.batch_norm1 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_1")
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.cat1 = tf.keras.layers.Concatenate(name="concatenate1")
        self.emb = tf.keras.layers.Embedding(
                len(vocabulary["article_id"]) + 1,
                config["article_embedding_dim"],
                name="article_embedding")
        self.gru = tf.keras.layers.GRU(config["gru_dim"])
        self.flatten = tf.keras.layers.Flatten()
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
        xgraphical = self.lookups["graphical_appearance_name"].lookup(
                inputs["article_id_hist"])
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourmaster = self.lookups["perceived_colour_master_name"].lookup(
                inputs["article_id_hist"])
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        #xprice = inputs["price"]
        #xprice = self.price_norm(xprice)
        #xpricemask = inputs["price_mask"]
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster,
            #xprice,
            #xpricemask,
            ])
        x = self.batch_norm0(x)
        x = self.gru(x)
        xclub = self.lookups["club_member_status"].lookup(
                inputs["customer_id"])
        xclub = self.club_encoder(xclub)
        xclub = self.flatten(xclub)
        xnews = self.lookups["fashion_news_frequency"].lookup(
                inputs["customer_id"])
        xnews = self.news_encoder(xnews)
        xnews = self.flatten(xnews)
        x = self.cat1([x, xclub, xnews])
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
                config,
                name="sequential_query_model")
        self.article_model = ArticleModel(
                vocabulary,
                lookups,
                config,
                name="article_model")
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
