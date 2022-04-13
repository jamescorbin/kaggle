import sys
import os
import pandas as pd

def get_unknown_article():
    return {
            "product_code": 0,
            "prod_name": b"",
            "product_type_no": 0,
            "product_type_name": b"",
            "product_group_name": b"",
            "graphical_appearance_no": 0,
            "graphical_appearance_name": b"",
            "colour_group_code": 0,
            "colour_group_name": b"",
            "perceived_colour_value_id": 0,
            "perceived_colour_value_name": b"",
            "perceived_colour_master_id": 0,
            "perceived_colour_master_name": b"",
            "department_no": 0,
            "department_name": b"",
            "index_code": 0,
            "index_name": b"",
            "index_group_no": 0,
            "index_group_name": b"",
            "section_no": 0,
            "section_name": b"",
            "garment_group_no": 0,
            "garment_group_name": b"",
            "detail_desc": b""}

def load_articles_ds():
    articles_fn = os.path.abspath(os.path.join(
            __file__, os.pardir,
            "data", "articles.csv"))
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
    articles_ds["detail_desc"] = articles_ds["detail_desc"].fillna("")
    articles_ds["article_id"] = articles_ds["article_id"].astype(str)
    for col in articles_bytes_cols:
        articles_ds[col] = (
            articles_ds[col].str.casefold()
                .str.encode("utf-8"))
    articles_ds = articles_ds.set_index("article_id")
    articles_ds.loc[b"0"] = get_unknown_article()
    return articles_ds

def load_customers_ds():
    customer_fn = os.path.abspath(os.path.join(
            __file__, os.pardir,
            "data", "customers.csv"))
    customers_bytes_cols = [
            "customer_id",
            "club_member_status",
            "fashion_news_frequency",]
    customers_ds = pd.read_csv(customer_fn)
    customers_ds["fashion_news_frequency"] = (
            customers_ds["fashion_news_frequency"].fillna(""))
    customers_ds["club_member_status"] = (
            customers_ds["club_member_status"].fillna(""))
    for col in customers_bytes_cols:
        customers_ds[col] = customers_ds[col].str.encode("utf-8")
    customers_ds.set_index("customer_id", inplace=True)
    return customers_ds

def append_previous_purchases(
        transaction_ds: pd.DataFrame,
        window: int=5,
        ) -> pd.DataFrame:
    article_id = "article_id"
    price = "price"
    sales_channel_id = "sales_channel_id"
    transaction_ds.sort_values("t_dat", inplace=True)
    for n in range(1, window + 1):
        transaction_ds[f"{article_id}_{n}"] = (
                transaction_ds
                    .groupby("customer_id")
                    [article_id]
                    .shift(n)
                    .fillna(b"0")
                    )
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
                    .fillna(0)
                    .astype(int))
    return transaction_ds

def load_transactions_ds():
    transact_fn = os.path.abspath(os.path.join(
            __file__, os.pardir,
            "data", "transactions_train.csv"))
    transactions_ds = pd.read_csv(transact_fn, nrows=1000)
    transactions_ds["article_id"] = transactions_ds["article_id"].astype(str)
    transactions_ds["article_id"] = (
            transactions_ds["article_id"].str.encode("utf-8"))
    transactions_ds = append_previous_purchases(transactions_ds)
    return transactions_ds
