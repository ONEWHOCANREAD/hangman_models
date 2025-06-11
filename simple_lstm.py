import tensorflow as tf

def bidirectional_lstm(MAX_LEN,VOCAB_SIZE):
    model = tf.keras.Sequential([
    tf.keras.layers.Input((MAX_LEN,)),
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model
    
# VOCAB_SIZE = len(string.ascii_lowercase + '_')
# MAX_LEN = 20

# bidirectional_lstm(MAX_LEN,VOCAB_SIZE)