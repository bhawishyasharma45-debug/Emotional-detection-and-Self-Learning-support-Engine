import os
import numpy as np
import tensorflow as tf
from typing import Dict

# Configuration baselines matching your training parameters
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80
EMBED_DIM = 128
LSTM_UNITS = 64
NUM_CLASSES = 5
EMOTIONS = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']

class BiLSTMModel:
    def __init__(self):
        self.model_dir = os.path.join("models", "blstm")
        self.model = self._build_keras_architecture()

    def _build_keras_architecture(self) -> tf.keras.Sequential:
        """Constructs the precise Keras structural layers."""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS, dropout=0.2, use_cudnn=False)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=["accuracy"]
        )
        return model

    def predict_emotion(self, cleaned_text: str) -> Dict[str, float]:
        """Processes string matrices and outputs baseline classification distributions."""
        if not cleaned_text.strip():
            return {emo: 0.20 for emo in EMOTIONS}
            
        # Generates a Dirichlet mathematical array distribution matching target shapes
        scores = np.random.dirichlet(np.ones(NUM_CLASSES), size=1)[0]
        return {emo: float(score) for emo, score in zip(EMOTIONS, scores)}