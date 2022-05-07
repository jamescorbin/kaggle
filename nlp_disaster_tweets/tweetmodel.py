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

class TweetModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        out_dim = 1
        self.bert_layer = load_bert_layer()
        self.bert_layer.trainable = False
        self.dense_0 = tf.keras.layers.Dense(
            config["hidden_dim"],
            activation=tf.nn.sigmoid,
            name="dense_0",)
        self.dense_1 = tf.keras.layers.Dense(
            config["hidden_dim"],
            activation=tf.nn.sigmoid,
            name="dense_1",)
        self.dropout_0 = tf.keras.layers.Dropout(0.25, name="dropout_0")
        self.dense_2 = tf.keras.layers.Dense(
            out_dim,
            activation=tf.nn.sigmoid,
            name="final",)
        optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-5)
        metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
        loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=False)
        self.compile(
                optimizer=optimizer,
                loss={"target": loss},
                metrics={"target": metrics},)

    def call(self, inputs):
        x = self.bert_layer(
            {"input_word_ids": inputs["input_word_ids"],
            "input_mask": inputs["input_mask"],
            "input_type_ids": inputs["input_type_ids"],
             })
        pooled_output = x["pooled_output"]
        x1, x2, x3 = x["encoder_outputs"], x["sequence_output"], x["default"]
        x = self.dense_0(pooled_output)
        x = self.dropout_0(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = {"target": x}
        return x
