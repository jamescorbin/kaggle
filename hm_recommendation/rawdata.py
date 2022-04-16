import sys
import os
import logging
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(name=__name__)
unk = "[unk]"

default_articles_fn = os.path.abspath(os.path.join(
        __file__, os.pardir,
        "data", "articles.csv"))
default_customer_fn = os.path.abspath(os.path.join(
        __file__, os.pardir,
        "data", "customers.csv"))
default_transact_fn = os.path.abspath(os.path.join(
        __file__, os.pardir,
        "data", "transactions_train.csv"))

def get_unknown_article():
    return {
            "product_code": 0,
            "prod_name": unk.encode("utf-8"),
            "product_type_no": 0,
            "product_type_name": unk.encode("utf-8"),
            "product_group_name": unk.encode("utf-8"),
            "graphical_appearance_no": 0,
            "graphical_appearance_name": unk.encode("utf-8"),
            "colour_group_code": 0,
            "colour_group_name": unk.encode("utf-8"),
            "perceived_colour_value_id": 0,
            "perceived_colour_value_name": unk.encode("utf-8"),
            "perceived_colour_master_id": 0,
            "perceived_colour_master_name": unk.encode("utf-8"),
            "department_no": 0,
            "department_name": unk.encode("utf-8"),
            "index_code": 0,
            "index_name": unk.encode("utf-8"),
            "index_group_no": 0,
            "index_group_name": unk.encode("utf-8"),
            "section_no": 0,
            "section_name": unk.encode("utf-8"),
            "garment_group_no": 0,
            "garment_group_name": unk.encode("utf-8"),
            "detail_desc": unk.encode("utf-8")}

def load_articles_ds(articles_fn: str=default_articles_fn) -> pd.DataFrame:
    logger.info(f"Opening articles dataset")
    articles_bytes_cols = [
            "article_id",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
            "detail_desc",
            ]
    articles_ds = pd.read_csv(articles_fn)
    articles_ds["article_id"] = (articles_ds["article_id"]
            .apply(lambda x: f"{x:010d}"))
    for col in articles_bytes_cols:
        articles_ds[col] = (
            articles_ds[col]
                .fillna(unk)
                .str.casefold()
                .str.encode("utf-8"))
    articles_ds = articles_ds.set_index("article_id")
    articles_ds.loc[b""] = get_unknown_article()
    return articles_ds

def load_customers_ds(customers_fn: str=default_customer_fn) -> pd.DataFrame:
    logger.info(f"Opening customers dataset")
    customers_bytes_cols = [
            "customer_id",
            "club_member_status",
            "fashion_news_frequency",]
    customers_ds = pd.read_csv(customers_fn)
    customers_ds["fashion_news_frequency"] = (
            customers_ds["fashion_news_frequency"]
                .fillna(unk)
                .str.casefold())
    customers_ds["club_member_status"] = (
            customers_ds["club_member_status"]
                .fillna(unk)
                .str.casefold())
    for col in customers_bytes_cols:
        customers_ds[col] = (customers_ds[col]
                             .str.encode("utf-8"))
    idx = pd.notnull(customers_ds["FN"])
    customers_ds.loc[idx, "fn_mask"] = 1.0
    customers_ds.loc[~idx, "fn_mask"] = 0.0
    customers_ds["FN"] = customers_ds["FN"].fillna(0.0)
    idx = pd.notnull(customers_ds["Active"])
    customers_ds.loc[idx, "active_mask"] = 1.0
    customers_ds.loc[~idx, "active_mask"] = 0.0
    customers_ds["Active"] = customers_ds["Active"].fillna(0.0)
    idx = pd.notnull(customers_ds["age"])
    customers_ds.loc[idx, "age_mask"] = 1.0
    customers_ds.loc[~idx, "age_mask"] = 0.0
    customers_ds["age"] = customers_ds["age"].fillna(0.0)
    customers_ds.set_index("customer_id", inplace=True)
    return customers_ds

def append_previous_purchases(
        transaction_ds: pd.DataFrame,
        window: int=5,
        ) -> pd.DataFrame:
    article_id = "article_id"
    price = "price"
    sales_channel_id = "sales_channel_id"
    for n in range(1, window + 1):
        transaction_ds[f"{article_id}_{n}"] = (
                transaction_ds
                    .groupby("customer_id")
                    [article_id]
                    .shift(n)
                    .fillna(b""))
        transaction_ds[f"{price}_{n}"] = (
                transaction_ds
                    .groupby("customer_id")
                    [price]
                    .shift(n))
        transaction_ds[f"{sales_channel_id}_{n}"] = (
                transaction_ds
                    .groupby("customer_id")
                    [sales_channel_id]
                    .shift(n)
                    .fillna(unk.encode("utf-8")))
    for n in range(1, window + 1):
        transaction_ds[f"{price}_{n}_mask"] = (
            pd.notnull(transaction_ds[f"{price}_{n}"]))
        transaction_ds[f"{price}_{n}"] = (
            transaction_ds[f"{price}_{n}"].fillna(0.0))
    return transaction_ds

def load_transactions_ds(
        transactions_fn: str=default_transact_fn,
        ) -> pd.DataFrame:
    logger.info(f"Opening transactions dataset")
    transactions_ds = pd.read_csv(transactions_fn, nrows=1000000)
    transactions_ds["article_id"] = (
            transactions_ds["article_id"]
                .apply(lambda x: f"{x:010d}")
                .str.encode("utf-8"))
    transactions_ds["t_dat"] = transactions_ds["t_dat"].str.encode("utf-8")
    transactions_ds["customer_id"] = (
            transactions_ds["customer_id"].str.encode("utf-8"))
    transactions_ds["sales_channel_id"] = (
            transactions_ds["sales_channel_id"]
                .apply(lambda x: str(x).encode("utf-8")))
    transactions_ds = append_previous_purchases(transactions_ds)
    return transactions_ds

def write_vocabulary(
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame,
        parent_dir: str="./vocabulary"):
    """
    """
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    article_ids = articles_ds.index.unique().tolist()
    garment_voc = articles_ds["garment_group_name"].unique().tolist()
    section_voc = articles_ds["section_name"].unique().tolist()
    index_voc = articles_ds["index_name"].unique().tolist()
    index_group_voc = articles_ds["index_group_name"].unique().tolist()
    department_voc = articles_ds["department_name"].unique().tolist()
    colour_value_voc = (articles_ds["perceived_colour_value_name"]
            .unique().tolist())
    colour_master_voc = (articles_ds["perceived_colour_master_name"]
            .unique().tolist())
    colour_group_voc = articles_ds["colour_group_name"].unique().tolist()
    graphical_appearance_voc = (articles_ds["graphical_appearance_name"]
            .unique().tolist())
    product_group_voc = (articles_ds["product_group_name"]
            .unique().tolist())
    product_type_voc = articles_ds["product_type_name"].unique().tolist()
    club_member_voc = customers_ds["club_member_status"].unique().tolist()
    fashion_news_voc = (customers_ds["fashion_news_frequency"]
            .unique().tolist())
    pairs = [
        (article_ids, "article_ids.txt"), (garment_voc, "garments.txt"),
        (section_voc, "sections.txt"),
        (index_voc, "index_names.txt"),
        (index_group_voc, "index_groups.txt"),
        (department_voc, "departments.txt"),
        (colour_value_voc, "colour_value_names.txt"),
        (colour_master_voc, "colour_master_names.txt"),
        (colour_group_voc, "colour_group_names.txt"),
        (graphical_appearance_voc, "graphical_appearance_names.txt"),
        (product_group_voc, "product_group_names.txt"),
        (product_type_voc, "product_type_names.txt"),
        (club_member_voc, "club_member_statuses.txt"),
        (fashion_news_voc, "fashion_news.txt"),
    ]
    for lst, bn in pairs:
        fn = os.path.join(parent_dir, bn)
        with open(fn, "w") as f:
            for w in lst:
                f.write(w.decode("utf-8") + "\n")

def load_vocabulary(
        parent_dir: str="./vocabulary"):
    """
    """
    vocabularies = {
        "article_ids": [],
        "garment_voc": [],
        "section_voc": [],
        "index_voc": [],
        "index_group_voc": [],
        "department_voc": [],
        "colour_value_voc": [],
        "colour_master_voc": [],
        "colour_group_voc": [],
        "graphical_appearance_voc": [],
        "product_group_voc": [],
        "product_type_voc": [],
        "club_member_voc": [],
        "fashion_news_voc": [],}
    pairs = [
        ("article_id", "article_ids.txt"),
        ("garment_group_name", "garments.txt"),
        ("section_name", "sections.txt"),
        ("index_name", "index_names.txt"),
        ("index_group_name", "index_groups.txt"),
        ("department_name", "departments.txt"),
        ("perceived_colour_value_name", "colour_value_names.txt"),
        ("perceived_colour_master_name", "colour_master_names.txt"),
        ("colour_group_name", "colour_group_names.txt"),
        ("graphical_appearance_name", "graphical_appearance_names.txt"),
        ("product_group_name", "product_group_names.txt"),
        ("product_type_name", "product_type_names.txt"),
        ("club_member_status", "club_member_statuses.txt"),
        ("fashion_news_frequency", "fashion_news.txt"),]
    for lst, bn in pairs:
        fn = os.path.join(parent_dir, bn)
        with open(fn, "r") as f:
            vocabularies[lst] = [w.strip().encode("utf-8") for w in f]
    return vocabularies

def vectorize_features(
        articles_ds: pd.DataFrame,
        customers_ds: pd.DataFrame,
        vocabulary: Dict[str, List[str]],
        ) -> Tuple[pd.DataFrame]:
    cols = ["garment_group_name",
            "section_name",
            "index_name",
            "index_group_name",
            "department_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "colour_group_name",
            "graphical_appearance_name",
            "product_group_name",
            "product_type_name",]
    for col in cols:
        dct = {x: i for i, x in enumerate(vocabulary[col])}
        articles_ds[col] = (
            articles_ds[col]
            .apply(lambda x: dct[x]))
    cols = ["club_member_status",
            "fashion_news_frequency"]
    for col in cols:
        dct = {x: i for i, x in enumerate(vocabulary[col])}
        customers_ds[col] = (
            customers_ds[col]
            .apply(lambda x: dct[x]))
    return articles_ds, customers_ds


