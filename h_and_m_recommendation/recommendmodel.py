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
        articles_ds = pd.read_parquet(config["articles_fn"])
        customers_fn = pd.read_parquet(config["customers_fn"])
        self.section_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                        dtype=tf.int64),
                    tf.constant(
                        articles_ds["section_name"].values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.section_name_lookup = tf.keras.layers.StringLookup(
                vocabulary=config["section_name"],
                name="section_name_lookup")
        self.section_encoder = tf.keras.layers.Embedding(
                config["section_name_dim"],
                config["section_name_dim"] // 4,
                name="section_encoder")
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
        self.article_id_embedding = tf.keras.layers.Embedding(
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
                config["graphical_appearance_name_dim"],
                config["graphical_appearance_name_dim"] // 2,
                name="graphical_encoder")
        self.colour_master_vec = tf.keras.layers.StringLookup(
                vocabulary=config["perceived_colour_master_name"],
                name="colour_master_vectorizer")
        self.colour_master_encoder = tf.keras.layers.Embedding(
                config["perceived_colour_master_name_dim"],
                config["perceived_colour_master_name_dim"] // 2,
                name="colour_master_encoder")
        self.batch_norm = tf.keras.layers.BatchNormalization(
                name="article_batch_norm")
        self.dropout_0 = tf.keras.layers.Dropout(
                0.20,
                name="dropout_0")
        self.cat = tf.keras.layers.Concatenate(name="concatenate")
        self.dense_0 = tf.keras.layers.Dense(
                units=config["factor_dim"],
                activation="linear",
                use_bias=False,
                name="dense_0")

    def call(self, inputs):
        x = inputs["article_id"]
        x = self.article_id_lookup(x)
        x = self.article_id_embedding(x)
        xsection = self.section_name_hashtable.lookup(inputs["article_id"])
        xsection = self.section_name_lookup(xsection)
        xsection = self.section_encoder(xsection)
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
            xsection,
            ])
        x = self.flatten(x)
        x = self.batch_norm(x)
        x = self.dropout_0(x)
        x = self.dense_0(x)
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
        articles_ds = pd.read_parquet(articles_fn)
        customers_ds = pd.read_parquet(customers_fn)
        self.section_name_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        articles_ds["article_id"].values,
                        dtype=tf.int64),
                    tf.constant(
                        articles_ds["section_name"].values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.section_name_lookup = tf.keras.layers.StringLookup(
                vocabulary=config["section_name"],
                name="section_name_lookup")
        self.section_encoder = tf.keras.layers.Embedding(
                config["section_name_dim"],
                config["section_name_dim"] // 4,
                name="section_encoder")
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
                        customers_ds["club_member_status"].values,
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.fashion_news_frequency_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        customers_ds["customer_id"].values,
                        dtype=tf.string),
                    tf.constant(
                        customers_ds["fashion_news_frequency"],
                        dtype=tf.string)),
                default_value=b"[UNK]")
        self.age_hashtable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(
                        customers_ds["customer_id"].values,
                        dtype=tf.string),
                    tf.constant(
                        customers_ds["age"],
                        dtype=tf.float32)),
                default_value=0.0)
        #self.price_norm = tf.keras.layers.Normalization(
        #        mean=config["price_mean"],
        #        variance=config["price_var"],
        #        name="price_norm")
        self.age_norm = tf.keras.layers.Normalization(
                mean=config["age_mean"],
                variance=config["age_var"],
                name="age_norm")
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
        self.batch_norm_0 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_0")
        self.dropout_0 = tf.keras.layers.Dropout(
                0.20,
                name="dropout_0",)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_1")
        self.dropout_1 = tf.keras.layers.Dropout(
                0.20,
                name="dropout_1",)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_2")
        self.dropout_2 = tf.keras.layers.Dropout(
                0.20,
                name="dropout_2",)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(
                name="sequential_batch_norm_3")
        self.dropout_3 = tf.keras.layers.Dropout(
                0.20,
                name="dropout_3",)
        self.cat = tf.keras.layers.Concatenate(name="concatenate_0")
        self.cat_1 = tf.keras.layers.Concatenate(name="concatenate_1")
        self.article_id_lookup = tf.keras.layers.IntegerLookup(
                vocabulary=config["article_id"],
                name="article_id_lookup")
        self.article_id_embedding = tf.keras.layers.Embedding(
                config["article_id_dim"],
                config["article_embedding_dim"],
                name="article_embedding")
        self.gru = tf.keras.layers.GRU(config["gru_dim"])
        self.flatten = tf.keras.layers.Flatten()
        self.dense_0 = tf.keras.layers.Dense(
                units=config["hidden_dim"],
                activation="sigmoid",
                use_bias=False,
                name="dense_0")
        self.dense_1 = tf.keras.layers.Dense(
                units=config["hidden_dim"],
                activation="sigmoid",
                use_bias=False,
                name="dense_1")
        self.output_0 = tf.keras.layers.Dense(
                units=config["factor_dim"],
                activation="linear",
                use_bias=False,
                name="output_0")

    def call(self, inputs):
        x = inputs["article_id_hist"]
        x = self.article_id_lookup(x)
        x = self.article_id_embedding(x)
        xsection = self.section_name_hashtable.lookup(inputs["article_id_hist"])
        xsection = self.section_name_lookup(xsection)
        xsection = self.section_encoder(xsection)
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
        x = self.cat([
            x,
            xgroup,
            xgraphical,
            xcolourmaster,
            xsection,
            #xprice,
            ])
        x = self.batch_norm_0(x)
        x = self.dropout_0(x)
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
        xage = self.age_hashtable.lookup(inputs["customer_id"])
        xage = self.age_norm(xage)
        x = self.cat_1([x, xclub, xnews, xage])
        x = self.batch_norm_1(x)
        x = self.dropout_1(x)
        x = self.dense_0(x)
        x = self.batch_norm_2(x)
        x = self.dropout_2(x)
        x = self.dense_1(x)
        x = self.batch_norm_3(x)
        x = self.dropout_3(x)
        x = self.output_0(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
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
        return self(features)

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
        return config
