import sys
import os
from typing import List
import re
import itertools
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text
import official.nlp.bert
import official.nlp.bert.tokenization

UNK = "UNK"
NUM = "number"
AT = "recipient"
http = "http"
html = "html"

def prepare_sentences(
        df: pd.DataFrame,
        stopwords: List[str]) -> pd.DataFrame:
    col = "text"
    df["text"] = df["text"].str.casefold()
    reg_hash_full = re.compile("(#)\w+")
    reg_hash = re.compile("(#)")
    reg_at = re.compile("(@)")
    reg_at_full = re.compile("(@)\w+")
    reg_href_full = re.compile("(htt)\S+")
    reg_html = re.compile("(&)\w+(;)")
    reg_x89 = re.compile(b"\xc2\x89".decode('utf-8')+"\S+")
    reg_special = re.compile("[^\w\s]")
    reg_contraction = re.compile("\s(s|m|t|(nt)|(ve)|w)\s")
    reg_numerals = re.compile("\d+[\s\d]*")
    reg_whitespace = re.compile("\s+")
    stop_str = ("(\s" + "\s)|(\s".join(stopwords) + "\s)"
            "(\A" + "\s)|(\A".join(stopwords) + "\s)"
            "(\s" + "\Z)|(\s".join(stopwords) + "\Z)")
    reg_stopwords = re.compile(stop_str)
    f = lambda x: [y.group() for y in reg_hash_full.finditer(x)]
    g = lambda x: ' '.join(x)
    df["hashtag"] = df["text"].apply(f).apply(g)
    df[col] = df[col].str.replace(reg_hash, ' ')
    f = lambda x: [y.group() for y in reg_at_full.finditer(x)]
    g = lambda x: ' '.join(x)
    df["at"] = df[col].apply(f).apply(g)
    df[col] = df[col].str.replace(reg_at_full, f" {AT} ")
    f = lambda x: len(list(reg_href_full.finditer(x)))
    df["href"] = df[col].apply(f)
    df[col] = df[col].str.replace(reg_href_full, f' {http} ')
    df[col] = df[col].str.replace(reg_html, f' {html} ')
    df[col] = df[col].str.replace(reg_x89, ' ')
    df[col] = df[col].str.replace(reg_special, ' ')
    df[col] = df[col].str.replace('_', " ")
    df[col] = df[col].str.replace(reg_contraction, ' ')
    df[col] = df[col].str.replace(reg_numerals, f' {NUM} ')
    df[col] = df[col].str.replace(reg_stopwords, " ")
    df[col] = df[col].str.replace(reg_whitespace, " ")
    df[col] = df[col].str.strip()
    return df

def write_vocabulary(
        ds: pd.DataFrame,
        vocab_fn: str="data/vocabulary.txt"):
    vocab = ds["text"].tolist()
    vocab = set(itertools.chain.from_iterable(
            [x.split() for x in vocab]))
    vocab = vocab | set([
        "[UNK]", "[MASK]",
        "[CLS]", "[SEP]",
        "[RANDOM]",])
    with open(vocab_fn, "w") as f:
        for word in vocab:
            f.write(f"{word}\n")
