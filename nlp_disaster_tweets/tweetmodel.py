import sys
import os
import tensorflow as tf
import tensorflow_text as tf_text
import load

def load_bert_layer():
    model_dir = os.path.join(
            os.getenv("HOME"),
            "model_repository",
            "all_bert_models",
            "small_bert_bert_en_uncased_L-4_H-256_A-4_2")
    #bert_layer = tf.keras.models.load_model(model_dir)
    bert_layer = tf.saved_model.load(model_dir)
    return bert_layer

def build_two_layer_model():
    out_dim = 1
    input_0 = tf.keras.Input(sequence_length,
                            dtype=tf.string,
                            name="words",)
    tokens = tf.text.WhitespaceTokenizer(input_0)

    #input1 = tf.keras.Input(sequence_length,

    #                        dtype=tf.dtypes.int32,
    #                        name="masks",)
    input2 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="segment_ids",)
    inputs = [input0, input1, input2]
    embed0 = tf.keras.layers.Embedding(
            num_unique_words,
            embed_dim,
            input_length=sequence_length,
            name="word_embedding",)
    lstm0 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units_0,
                name="lstm_0",
                return_sequences=True,))
    lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units_1,
                name="lstm_1",))
    dense0 = tf.keras.layers.Dense(
            out_dim,
            activation=tf.nn.sigmoid,
            name="final",)

def build_bert_model(config):
    out_dim = 1
    sequence_length = config["max_sequence_length"]
    input_0 = tf.keras.Input(
        1,
        dtype=tf.string,
        name="text")
    inputs = [input_0]
    tokenizer, vocabulary = load.get_tokenizer()
    _START_TOKEN = vocabulary.index(b"[CLS]")
    _END_TOKEN = vocabulary.index(b"[SEP]")
    _MASK_TOKEN = vocabulary.index(b"[MASK]")
    _UNK_TOKEN = vocabulary.index(b"[UNK]")
    trimmer = tf_text.WaterfallTrimmer(sequence_length)
    x = inputs[0]
    _f0 = lambda x: tokenizer.tokenize(x).merge_dims(-2, -1)
    lambda_0 = tf.keras.layers.Lambda(_f0)
    x = lambda_0(x)
    _f1 = lambda x: trimmer.trim([x])[0]
    lambda_1 = tf.keras.layers.Lambda(_f1)
    x = lambda_1(x)
    _f2 = (lambda x: tf_text.combine_segments(
            x,
            start_of_sequence_id=_START_TOKEN,
            end_of_segment_id=_END_TOKEN))
    lambda_2 = tf.keras.layers.Lambda(_f2)
    x, type_ids = lambda_2([x])
    _f3 =(
        lambda x: tf_text.pad_model_inputs(
            x,
            max_seq_length=sequence_length))
    lambda_3 = tf.keras.layers.Lambda(_f3)
    word_ids, input_mask = lambda_3(x)
    type_ids, _ = lambda_3(type_ids)
    bert_layer = load_bert_layer()
    dense_0 = tf.keras.layers.Dense(
            config["hidden_dim"],
            activation=tf.nn.relu,
            name="dense_0",)
    dropout_0 = tf.keras.layers.Dropout(0.5, name="dropout_0")
    dense_1 = tf.keras.layers.Dense(
            out_dim,
            activation=tf.nn.sigmoid,
            name="final",)
    pooled, sequence = bert_layer(
            word_ids,
            input_mask,
            type_ids)
    clf_output = sequence[:, 0, :]
    x = dense_0(clf_output)
    x = dropout_0(x)
    x = dense_1(x)
    model = tf.keras.Model(
            inputs=inputs,
            outputs=[x],
            name="bert_model",)
    optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-5)
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False)
    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,)
    return model


class TweetModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        out_dim = 1
        self.bert_layer = load_bert_layer()
        self.dense_0 = tf.keras.layers.Dense(
            config["hidden_dim"],
            activation=tf.nn.relu,
            name="dense_0",)
        self.dropout_0 = tf.keras.layers.Dropout(0.5, name="dropout_0")
        self.dense_1 = tf.keras.layers.Dense(
            out_dim,
            activation=tf.nn.sigmoid,
            name="final",)
        optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-5)
        metrics = [tf.keras.metrics.BinaryAccuracy()]
        loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=False)
        self.compile(
                optimizer=optimizer,
                loss={"target": loss},
                metrics={"target": metrics},)

    def call(self, inputs):
        word_ids = inputs["word_ids"]
        input_mask = inputs["input_mask"]
        type_ids = inputs["type_ids"]
        pooled, sequence = self.bert_layer(
            {"input_word_ids": inputs["word_ids"],
            "input_mask": inputs["input_mask"],
            "input_type_ids": inputs["type_ids"],
             })
        clf_output = sequence[:, 0, :]
        x = self.dense_0(clf_output)
        x = self.dropout_0(x)
        x = self.dense_1(x)
        x = {"target": x}
        return x

