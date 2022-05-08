import sys
import os
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)

def get_tokenizer():
    """
    """
    folder_bert = os.path.join(os.getenv("HOME"),
            "model_repository",
            "all_bert_models",
            #"uncased_L-4_H-256_A-4",
            "bert_en_uncased_preprocess_3",
            )
    vocab_file = os.path.join(
            folder_bert,
            "assets",
            "vocab.txt")
    if True:
        vocabulary = []
        with open(vocab_file, "r") as f:
            vocabulary = [x.strip().encode("utf-8")
                    for x in f if x.strip() != ""]
        tokenizer = tf.keras.models.load_model(folder_bert)
    else:
        tokenizer = tf_text.BertTokenizer(
                vocab_file,
                token_out_type=tf.int64)
        with open(vocab_file, "r") as f:
            vocabulary = [x.strip().encode("utf-8")
                    for x in f if x.strip() != ""]
    return tokenizer, vocabulary

def encode_dataset(ds, config, tokenizer, vocabulary):
    _START_TOKEN = vocabulary.index(b"[CLS]")
    _END_TOKEN = vocabulary.index(b"[SEP]")
    max_sequence_length = config["max_sequence_length"]
    max_selections_per_batch = config["max_selections_per_batch"]
    _f0 = lambda x: {
            "text": x["text"],
            "input_word_ids": tokenizer
                    .tokenize(x["text"])
                    .merge_dims(-2, -1),}
    ds = ds.map(_f0)
    trimmer = tf_text.WaterfallTrimmer(max_sequence_length)
    _f1 = lambda x: {
            "text": x["text"],
            "input_word_ids": trimmer.trim([x["input_word_ids"]])[0],}
    ds = ds.map(_f1)
    _f2 = lambda x: tf_text.combine_segments(
            [x["input_word_ids"]],
            start_of_sequence_id=_START_TOKEN,
            end_of_segment_id=_END_TOKEN)
    _f3 = lambda x: {
            "text": x["text"],
            "input_word_ids": _f2(x)[0],
            "input_type_ids": _f2(x)[1],
            }
    ds = ds.map(_f3)
    return ds

def pad_sequence(
        ds,
        max_sequence_length):
    _f4 = lambda x: tf_text.pad_model_inputs(
            x,
            max_seq_length=max_sequence_length)
    _f5 = lambda x: {
            "text": x["text"],
            "input_word_ids": tf.squeeze(_f4(x["input_word_ids"])[0]),
            "input_mask": tf.squeeze(_f4(x["input_word_ids"])[1]),
            "input_type_ids": tf.squeeze(_f4(x["input_type_ids"])[0]),
            #"masked_lm_ids": x["masked_lm_ids"],
            #"masked_lm_pos": x["masked_lm_pos"],
            #"masked_ids": x["masked_ids"],
            }
    ds = ds.map(_f5)
    return ds

def augment_dataset(ds: tf.data.Dataset,
            vocabulary,
            max_selections_per_batch,
            ):
    len_vocabulary = len(vocabulary)
    _START_TOKEN = vocabulary.index(b"[CLS]")
    _END_TOKEN = vocabulary.index(b"[SEP]")
    mask_token = vocabulary.index(b"[MASK]")
    _UNK_TOKEN = vocabulary.index(b"[UNK]")
    random_selector = tf_text.RandomItemSelector(
            max_selections_per_batch=max_selections_per_batch,
            selection_rate=0.2,
            unselectable_ids=[_START_TOKEN,
                              _END_TOKEN,
                              _UNK_TOKEN])
    mask_values_chooser = tf_text.MaskValuesChooser(
            len_vocabulary,
            mask_token=mask_token,
            mask_token_rate=0.8)
    _f6 = lambda x: tf_text.mask_language_model(
            x["input_word_ids"],
            random_selector,
            mask_values_chooser)
    _f7 = lambda x: {
                    "text": x["text"],
                    "input_word_ids": _f6(x)[0],
                    "input_type_ids": x["input_type_ids"],
                    "masked_pos": _f6(x)[1],
                    "masked_ids": _f6(x)[2],
                    }
    ds = ds.map(_f7)
    _f9 = lambda x: tf_text.pad_model_inputs(
            x,
        max_seq_length=max_selections_per_batch)
    _f10 = lambda x: {
            "text": x["text"],
            "input_word_ids": x["input_word_ids"],
            "input_type_ids": x["input_type_ids"],
            "masked_lm_ids": _f9(x["masked_pos"])[0],
            "masked_lm_pos": _f9(x["masked_pos"])[1],
            "masked_ids": _f9(x["masked_ids"])[0],
            }
    ds = ds.map(_f10)
    return ds

def get_tfds(ds: pd.DataFrame, config):
    ds_x = tf.data.Dataset.from_tensor_slices(
            {"text": ds[["text"]].values})
    ds_y = tf.data.Dataset.from_tensor_slices(
            {"target": ds[["target"]].values})
    tokenizer, vocabulary = get_tokenizer()
    ds_x = encode_dataset(ds_x, config, tokenizer, vocabulary)
    ds_x = augment_dataset(ds_x,
            vocabulary,
            config["max_selections_per_batch"],)
    ds_x = pad_sequence(ds_x, config["max_sequence_length"])
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    x_train = (ds
            .enumerate()
            .filter(lambda x, y:
                    x % config["split_mod_k"] <= config["train_k"])
            .map(lambda x, y: y)
            .shuffle(config["shuffle_buffer"]))
    x_valid = (ds
            .enumerate()
            .filter(lambda x, y:
                    (x % config["split_mod_k"] > config["train_k"])
                    & (x % config["split_mod_k"] <= config["valid_k"]))
            .map(lambda x, y: y))
    x_test = (ds
            .enumerate()
            .filter(lambda x, y:
                    x % config["split_mod_k"] > config["valid_k"])
            .map(lambda x, y: y))
    return x_train, x_valid, x_test
