import tensorflow as tf

def cnn_lstm_model(MAX_LEN, VOCAB_SIZE,EMBED_DIM=32):
    cnn_lstm_model = tf.keras.Sequential([
        tf.keras.layers.Input((MAX_LEN,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=False),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(26, activation='softmax')   # Predict one of 26 letters (a-z)
    ])

    cnn_lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #cnn_lstm_model.summary()
    return cnn_lstm_model

def cnn_lstm_model_with_attention(MAX_LEN, VOCAB_SIZE, EMBED_DIM=32):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=3e-4,
                    decay_steps=10_000,  # Adjust based on your dataset size
                    decay_rate=0.9,
                    staircase=True
                    )

    # 2. AdamW Optimizer with Weight Decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-5,       # Regularization
        beta_1=0.9,             # Momentum
        beta_2=0.999,           # RMSprop-like smoothing
        clipnorm=1.0,           # Gradient clipping
        ema_momentum=0.99       # Optional: Exponential Moving Average
        )



    """Build the enhanced CNN-LSTM model with attention"""
    # Character input
    char_input = tf.keras.Input(shape=(MAX_LEN,), name='char_input')
    
    # Embedding layer
    embedding = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE, 
        output_dim=32, 
    )(char_input)
    
    # Parallel CNN branches
    conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(embedding)
    conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(embedding)
    conv = tf.keras.layers.concatenate([conv1, conv2])
    
    # Bidirectional LSTM with attention
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(conv)
    
    # Attention mechanism
    attention = tf.keras.layers.Attention()([lstm, lstm])
    
    # Global pooling and dense layers
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(pooled)
    dropout = tf.keras.layers.Dropout(0.3)(dense1)
    output = tf.keras.layers.Dense(26, activation='softmax')(dropout)
    
    model = tf.keras.Model(inputs=char_input, outputs=output)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # For integer-encoded targets
        metrics=['accuracy'],
        weighted_metrics=['accuracy']  # Optional: if using sample weights
    )
    
    return model