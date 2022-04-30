"""
"""
import sys
import os
from typing import List, Dict, Any
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import tfsalesdata
import rawdata

class CustomerModel(tf.keras.Model):
    def __init__(self,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        self.emb = tf.keras.layers.Embedding(
                config["customer_id_dim"],
                config["factor_dim"])
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = inputs["customer_id"]
        x = self.emb(x)
        x = self.flatten(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
                "config": self.config,
                       })
        return config

class ArticleModel(tf.keras.Model):
    def __init__(self,
                 config: Dict[str, Any],
                 **kwargs):
        super().__init__(**kwargs)
        output_type = "one_hot"
        self.config = config
        articles_ds = pd.read_csv(config["articles_fn"])
        customers_fn = pd.read_csv(config["customers_fn"])
        #self.section_name_lookup = tf.lookup.StaticHashTable(
        #        tf.lookup.KeyValueTensorInitializer(
        #            tf.constant(lookups["section_name"][0], dtype=tf.int64),
        #            tf.constant(lookups["section_name"][1], dtype=tf.int64)),
        #        default_value=0)
        #self.section_encoder = tf.keras.layers.Embedding(
        #        config["section_name_dim"],
        #        config["section_name_dim"]) // 4,
        #        name="section_encoder") 
        self.product_group_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                        dtype=tf.int64),
                    tf.constant(
                        articles_ds["product_group_name"].values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.graphical_appearance_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                        dtype=tf.int64),
                    tf.constant(
                        articles_ds["graphical_appearance_name"].values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.perceived_colour_master_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                        dtype=tf.int64),
                    tf.constant(
                        articles_ds["perceived_colour_master_name"].values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.article_id_lookup = tf.keras.layers.IntegerLookup(
                vocabulary=config["article_id"],
                name="article_id_lookup")
        self.emb = tf.keras.layers.Embedding(
                config["article_id_dim"],
                config["article_embedding_dim"],
                name="article_id_embedding")
        self.flatten = tf.keras.layers.Flatten()
        self.product_group_name_lookup = tf.keras.layers.StringLookup(
                vocabulary=config["product_group_name"],
                name="product_group_name")
        self.group_encoder = tf.keras.layers.Embedding(
                config["product_group_name_dim"],
                config["product_group_name_dim"] // 2,
                name="product_group_encoder")
        self.graphical_vec = tf.keras.layers.StringLookup(
                vocabulary=config["graphical_appearance_name"],
                name="graphical_vectorizer")
        self.graphical_encoder = tf.keras.layers.Embedding(
                config["graphical_appearance_name_dim"] + 1,
                config["graphical_appearance_name_dim"] // 2,
                name="graphical_encoder")
        self.colour_master_vec = tf.keras.layers.StringLookup(
                vocabulary=config["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.Embedding(
                config["perceived_colour_master_name_dim"] + 1,
                config["perceived_colour_master_name_dim"] // 2,
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
        x = self.article_id_lookup(x)
        x = self.emb(x)
        #xsection = self.section_name_lookup.lookup(inputs["article_id"])
        #xsection = self.section_encoder(xsection)
        xgroup = self.product_group_name_hashtable.lookup(
                inputs["article_id"])
        xgroup = self.product_group_name_lookup(xgroup)
        xgroup = self.group_encoder(xgroup)
        xgraphical = self.graphical_appearance_name_hashtable.lookup(
                inputs["article_id"])
        xgraphical = self.graphical_vec(xgraphical)
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourmaster = self.perceived_colour_master_name_hashtable.lookup(
                inputs["article_id"])
        xcolourmaster = self.colour_master_vec(xcolourmaster)
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster,
            #xsection,
            ])
        x = self.flatten(x)
        x = self.batch_norm(x)
        x = self.dense0(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
                "config": self.config,
                })
        return config

class SequentialQueryModel(tf.keras.Model):
    def __init__(self,
                 config: Dict[str, Any],
                 **kwargs):
        super().__init__(**kwargs)
        articles_fn = config["articles_fn"]
        customers_fn = config["customers_fn"]
        articles_ds = pd.read_csv(articles_fn)
        customers_ds = pd.read_csv(customers_fn)
        #self.section_name_lookup = tf.lookup.StaticHashTable(
        #        tf.lookup.KeyValueTensorInitializer(
        #            tf.constant(lookups["section_name"][0],
        #                        dtype=tf.int64),
        #            tf.constant(lookups["section_name"][1],
        #                        dtype=tf.int64)),
        #        default_value=0)
        #self.section_encoder = tf.keras.layers.Embedding(
        #        config["section_name_dim"] + 1,
        #        config["section_name_dim"] // 4,
        #        name="section_encoder")
        self.product_group_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                                dtype=tf.int64),
                    tf.constant(
                        articles_ds["product_group_name"].values,
                                dtype=tf.string)),
                default_value=b"[UNK]")
        self.graphical_appearance_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                                dtype=tf.int64),
                    tf.constant(
                        articles_ds["graphical_appearance_name"].values,
                                dtype=tf.string)),
                default_value=b"[UNK]")
        self.perceived_colour_master_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                        dtype=tf.int64),
                    tf.constant(
                        articles_ds["perceived_colour_master_name"].values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.club_member_status_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        customers_ds["customer_id"].values,
                        dtype=tf.string),
                    tf.constant(
                        customers_ds["club_member_status"]
                            .fillna("[UNK]").values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.fashion_news_frequency_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        customers_ds["customer_id"].values,
                        dtype=tf.string),
                    tf.constant(
                        customers_ds["fashion_news_frequency"]
                            .fillna("[UNK]").values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        #self.price_norm = tf.keras.layers.Normalization(
        #        mean=config["price_mean"],
        #        variance=config["price_var"],
        #        name="price_norm")
        self.product_group_name_lookup = tf.keras.layers.StringLookup(
                vocabulary=config["product_group_name"],
                name="product_group_name")
        self.group_encoder = tf.keras.layers.Embedding(
                config["product_group_name_dim"],
                config["product_group_name_dim"] // 2,
                name="product_group_encoder")
        self.graphical_vec = tf.keras.layers.StringLookup(
                vocabulary=config["graphical_appearance_name"],
                name="graphical_vectorizer")
        self.graphical_encoder = tf.keras.layers.Embedding(
                config["graphical_appearance_name_dim"],
                config["graphical_appearance_name_dim"] // 2,
                name="graphical_encoder")
        #self.graphical_norm = tf.keras.layers.BatchNormalization()
        self.colour_master_vec = tf.keras.layers.StringLookup(
                vocabulary=config["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.Embedding(
                config["perceived_colour_master_name_dim"],
                config["perceived_colour_master_name_dim"] // 2,
                name="colour_master_encoder")
        #self.colour_master_norm = tf.keras.layers.BatchNormalization()
        self.club_member_status_lookup = tf.keras.layers.StringLookup(
                vocabulary=config["club_member_status"],
                name="club_member_status_lookup")
        self.club_encoder = tf.keras.layers.Embedding(
                config["club_member_status_dim"],
                config["club_member_status_dim"] - 1,
                name="club_encoder")
        self.fashion_news_frequency_lookup = tf.keras.layers.StringLookup(
                vocabulary=config["fashion_news_frequency"],
                name="fashion_news_frequency_lookup")
        self.news_encoder = tf.keras.layers.Embedding(
                config["fashion_news_frequency_dim"],
                config["fashion_news_frequency_dim"] - 1,
                name="news_frequency_encoder")
        self.batch_norm0 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_0")
        self.batch_norm1 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_1")
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.cat1 = tf.keras.layers.Concatenate(name="concatenate1")
        self.article_id_lookup = tf.keras.layers.IntegerLookup(
                vocabulary=config["article_id"],
                name="article_id_lookup")
        self.emb = tf.keras.layers.Embedding(
                config["article_id_dim"],
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
        x = self.article_id_lookup(x)
        x = self.emb(x)
        #xsection = self.section_name_lookup.lookup(inputs["article_id_hist"])
        #xsection = self.section_encoder(xsection)
        xgroup = self.product_group_name_hashtable.lookup(
                inputs["article_id_hist"])
        xgroup = self.product_group_name_lookup(xgroup)
        xgroup = self.group_encoder(xgroup)
        xgraphical = self.graphical_appearance_name_hashtable.lookup(
                inputs["article_id_hist"])
        xgraphical = self.graphical_vec(xgraphical)
        xgraphical = self.graphical_encoder(xgraphical)
        xcolourmaster = self.perceived_colour_master_name_hashtable.lookup(
                inputs["article_id_hist"])
        xcolourmaster = self.colour_master_vec(xcolourmaster)
        xcolourmaster = self.colour_master_encoder(xcolourmaster)
        #xprice = inputs["price"]
        #xprice = self.price_norm(xprice)
        #xpricemask = inputs["price_mask"]
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster,
            #xsection,
            #xprice,
            #xpricemask,
            ])
        x = self.batch_norm0(x)
        x = self.gru(x)
        xclub = self.club_member_status_hashtable.lookup(
                inputs["customer_id"])
        xclub = self.club_member_status_lookup(xclub)
        xclub = self.club_encoder(xclub)
        xclub = self.flatten(xclub)
        xnews = self.fashion_news_frequency_hashtable.lookup(
                inputs["customer_id"])
        xnews = self.fashion_news_frequency_lookup(xnews)
        xnews = self.news_encoder(xnews)
        xnews = self.flatten(xnews)
        x = self.cat1([x, xclub, xnews])
        x = self.batch_norm1(x)
        x = self.dense0(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
                "config": self.config,
                })
        return config

class RetrievalModel(tfrs.Model):
    def __init__(self,
                 config: Dict[str, List[Any]],
                 **kwargs):
        super().__init__(**kwargs)
        articles_tf = tfsalesdata.make_articles_tf(
            rawdata.load_articles_ds(config["articles_fn"]))
        self.config = config
        self.customer_model = SequentialQueryModel(
                config,
                name="sequential_query_model")
        self.article_model = ArticleModel(
                config,
                name="article_model")
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                articles_tf
                    .batch(config["top_k_batch_size"])
                    .map(self.article_model)))
        opt = tf.keras.optimizers.Adagrad(learning_rate=0.1)
        self.compile(optimizer=opt)

    def call(self, features):
        return self.task(
            self.customer_model(features),
            self.article_model(features))

    def compute_loss(self, features, training=False) -> tf.Tensor:
        #return self.task(
        #    self.customer_model(features),
        #    self.article_model(features))
        return self(features)

    def get_config(self):
        config = super().get_config()
        config.update({
                "config": self.config,
                })
        return config
