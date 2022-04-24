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

    def get_config(self):
        config = super().get_config()
        config.update({"vocabulary": self.vocabulary,
                       "lookups": self.lookups,
                        "config": self.config,
                       })
        return config

class ArticleModel(tf.keras.Model):
    def __init__(self,
                 vocabulary,
                 lookups,
                 config: Dict[str, Any],
                 **kwargs):
        super().__init__(**kwargs)
        output_type = "one_hot"
        self.vocabulary = vocabulary
        self.lookups = lookups
        self.config = config
        self.section_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["section_name_lookup"][0]),
                    tf.constant(lookup_pairs["section_name_lookup"][1])),
                default_value=0)
        self.product_group_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["product_group_name"][0]),
                    tf.constnat(lookup_pairs["product_group_name"][1])),
                default_value=0)
        self.graphical_appearance_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["graphical_appearance_name"][0]),
                    tf.constant(lookup_pairs["graphical_appearance_name"][1])),
                default_value=0)
        self.perceived_colour_master_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["perceived_colour_master_name"][0]),
                    tf.constant(lookup_pairs["perceived_colour_master_name"][1])),
                default_value=0)
        self.emb = tf.keras.layers.Embedding(
        self.perceived_colour_master_name_lookup = tf.lookup.StaticHashTable(
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
        xsection = self.secion_name_lookup.lookup(
                inputs["article_id"])
        xsection = self.section_encoder(xsection)
        xgroup = self.product_group_name_lookup.lookup(
                inputs["article_id"])
        xgroup = self.group_encoder(xgroup)
        xgraphical = self.graphical_appearance_name_lookup.lookup(
                inputs["article_id"])
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourmaster = self.perceived_colour_master_name_lookup.lookup(
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

    def get_config(self):
        config = super().get_config()
        config.update({"vocabulary": self.vocabulary,
                       "lookups": self.lookups,
                        "config": self.config,
                       })
        return config

class SequentialQueryModel(tf.keras.Model):
    def __init__(self,
                 vocabulary,
                 lookups,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        self.lookups = lookups
        self.section_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["section_name_lookup"][0]),
                    tf.constant(lookup_pairs["section_name_lookup"][1])),
                default_value=0)
        self.product_group_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["product_group_name"][0]),
                    tf.constnat(lookup_pairs["product_group_name"][1])),
                default_value=0)
        self.graphical_appearance_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["graphical_appearance_name"][0]),
                    tf.constant(lookup_pairs["graphical_appearance_name"][1])),
                default_value=0)
        self.perceived_colour_master_name_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["perceived_colour_master_name"][0]),
                    tf.constant(lookup_pairs["perceived_colour_master_name"][1])),
                default_value=0)
        self.club_member_status_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["club_member_status"][0]),
                    tf.constant(lookup_pairs["club_member_status"][1])),
                default_value=0)
        self.fashion_news_frequency_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(lookup_pairs["fashion_news_frequency"][0]),
                    tf.constant(lookup_pairs["fashion_news_frequency"][1])),
                default_value=0)
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
        xgroup = selfproduct_group_name_lookup.lookup(
                inputs["article_id_hist"])
        xgroup = self.group_encoder(xgroup)
        xgraphical = self.graphical_appearance_name_lookup.lookup(
                inputs["article_id_hist"])
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourmaster = self.perceived_colour_master_name_lookup.lookup(
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
        xclub = self.club_member_status_lookup.lookup(
                inputs["customer_id"])
        xclub = self.club_encoder(xclub)
        xclub = self.flatten(xclub)
        xnews = self.fashion_news_frequency_lookup.lookup(
                inputs["customer_id"])
        xnews = self.news_encoder(xnews)
        xnews = self.flatten(xnews)
        x = self.cat1([x, xclub, xnews])
        x = self.batch_norm1(x)
        x = self.dense0(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"vocabulary": self.vocabulary,
                       "lookups": self.lookups,
                        "config": self.config,
                       })
        return config

class RetrievalModel(tfrs.Model):
    def __init__(self,
                 vocabulary,
                 articles_ds,
                 lookups: Dict[str, List[Any]],
                 config: Dict[str, List[Any]],
                 **kwargs):
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
