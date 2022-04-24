
def bert_tokenize(text_arr: np.array):
    """
    Perform preprocessing required for the BERT model.

    Args:
        df (pandas.DataFrame): Dataframe to be transformed.
        col (str): Column in ${df} to be transformed.

    Returns:
        tokenizer (bert.tokenization.FullTokenizer): BERT tokenizer
            downloaded and fit to the dataframe.
        num_unique_words (int): Number of unique words -- fixed for
            BERT tokenizer.
        word_cols (list(str)): Columns for encoded words in ${df}.
        mask_cols (list(str)): Columns for word masking in ${df}.
        type_cols (list(str)): Columns for word type in ${df}.
    """
    gs_folder_bert = ("gs://cloud-tpu-checkpoints"
                      "/bert/keras_bert/uncased_L-12_H-768_A-12")
    vocab_file = os.path.join(gs_folder_bert, "vocab.txt")
    bdry = ["[CLS] ", " [SEP]"]
    tf.io.gfile.listdir(gs_folder_bert)
    tokenizer = official.nlp.bert.tokenization.FullTokenizer(
                    vocab_file=vocab_file,
                    do_lower_case=True)
    _f = lambda x: bdry[0] + x + bdry[1]
    text_arr = _f(text_arr)
    bert_token = (lambda x: tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(x)))
    num_unique_words = len(tokenizer.vocab)
    words_ids = tf.keras.preprocessing.sequence.pad_sequences(
                                    df[col].apply(bert_token))
    _f = lambda x: [1]*(len(tokenizer.tokenize(x)))
    masks = tf.keras.preprocessing.sequence.pad_sequences(
                                        df[col].apply(_f))
    type_ids = np.zeros(words_ids.shape, dtype=np.int32)
    return tokenizer, num_unique_words, words_ids, masks, type_ids

def tf_tokenizer(
                text_arr: np.array,
                 num_unique_words: Optional[int]=None,
                 ):
    bdry = []
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
                                    num_words=num_unique_words,
                                    oov_token="unk",)
    tokenizer.fit_on_texts(text_arr)
    config = tokenizer.get_config()
    num_unique_words = len(config["index_word"])
    prewords = tokenizer.texts_to_sequences(text_arr)
    index_word = json.loads(config["index_word"])
    logger.info("text_arr " + text_arr[0])
    logger.info("encoding" + str(prewords[0]))
    output = [index_word[str(i)] for i in prewords[0]]
    logger.info("translate" + str(output))
    words_ids = tf.keras.preprocessing.sequence.pad_sequences(prewords)
    premask = [[1 for i in arr] for arr in prewords]
    masks = tf.keras.preprocessing.sequence.pad_sequences(premask)
    type_ids = np.zeros(words_ids.shape, dtype=np.int32)
    return tokenizer, num_unique_words, words_ids, masks, type_ids

def load_pretrained_bert(url_bert: str=("https://tfhub.dev/tensorflow"
                                "/bert_en_uncased_L-12_H-768_A-12/2"),
                         ):
    bert_layer = hub.KerasLayer(url_bert, trainable=True)
    return bert_layer

def build_two_layer_model(sequence_length: int,
                          num_unique_words: int,
                          embed_dim: int,
                          units_0: int,
                          units_1: int,
                          ):
    out_dim = 1
    input0 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="words_ids",)
    input1 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="masks",)
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    #loss = tf.keras.losses.KLDivergence()
    x = input0
    x = embed0(x)
    x = lstm0(x)
    x = lstm1(x)
    x = dense0(x)
    model = tf.keras.Model(inputs=inputs, outputs=[x],
                           name="bi-directional")
    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,)
    return model

def build_conv_model(
			sequence_length: int,
			num_unique_words: int,
			embed_dim: int,
			filters: int,
			window: int,
			pool_size: int,
			units: int,
			dense_0_dim: int,
			):
    out_dim = 1
    input0 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="words_ids",)
    input1 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="masks",)
    input2 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="segment_ids",)
    embed0 = tf.keras.layers.Embedding(
            num_unique_words,
            embed_dim,
            input_length=sequence_length,
            name="word_embedding",)
    conv0 = tf.keras.layers.Conv1D(
            filters,
            window)
    pool0 = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size,)
    flat0 = tf.keras.layers.Flatten(name="flat_0")
    lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units,
                name="lstm_1",))
    dense0 = tf.keras.layers.Dense(
            dense_0_dim,
            activation=tf.nn.sigmoid,
            name="dense_0",)
    dense1 = tf.keras.layers.Dense(
            out_dim,
            activation=tf.nn.sigmoid,
            name="dense_1",)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    #loss = tf.keras.losses.KLDivergence()
    x = input0
    x = embed0(x)
    x = conv0(x)
    x = pool0(x)
    x = flat0(x)
    x = dense0(x)
    x = dense1(x)
    model = tf.keras.Model(inputs=[input0], outputs=[x],
                           name="conv_model")
    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,)
    return model

def build_bert_model(bert_layer: tf.keras.layers.Layer,
                     sequence_length: int,
                     units_0: int):
    out_dim = 1
    input0 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="word_ids",)
    input1 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="masks",)
    input2 = tf.keras.Input(sequence_length,
                            dtype=tf.dtypes.int32,
                            name="segment_ids",)
    inputs = [input0, input1, input2]
    dense0 = tf.keras.layers.Dense(units_0,
                                   activation=tf.nn.relu,
                                   name="dense_0",)
    dropout0 = tf.keras.layers.Dropout(0.5)
    dense1 = tf.keras.layers.Dense(
            out_dim,
            activation=tf.nn.sigmoid,
            name="final",)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    pooled, sequence = bert_layer(inputs)
    clf_output = sequence[:, 0, :]
    x = dense0(clf_output)
    x = dropout0(x)
    x = dense1(x)
    model = tf.keras.Model(inputs=inputs,
                           outputs=[x],
                           name="bert_model",)
    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,)
    return model


