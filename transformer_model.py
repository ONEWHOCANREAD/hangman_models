import tensorflow as tf
#transformer model
MAX_LEN = 20               # Maximum word length
VOCAB_SIZE = 27            # 26 letters + '_' (mask)
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):  # <-- add `training=None`
        attn_output = self.att(inputs, inputs, attention_mask=None)
        attn_output = self.dropout1(attn_output, training=training)  # training used here
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)  # training used here
        return self.layernorm2(out1 + ffn_output)

    
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

def transformer_model(MAX_LEN, VOCAB_SIZE, EMBED_DIM=128, NUM_HEADS=4, FF_DIM=256):
    """
    Build a Transformer model for character-level classification.
    """
    # Input layer
    inputs = tf.keras.Input(shape=(MAX_LEN,))
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs)

    # Transformer block
    for _ in range(2):  # Stack multiple transformer blocks
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(26, activation="softmax")(x)  # Predict one of 26 letters

    transformer_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    transformer_model.summary()
    return transformer_model