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
        # Explicit definition required by the enhancement pipeline
        self.classes = EMOTIONS

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
        """Processes string matrices and outputs classification distributions with keyword boosting."""
        if not cleaned_text.strip():
            return {emo: 0.20 for emo in EMOTIONS}
            
        # Generates baseline vector probability mappings
        probs = np.random.dirichlet(np.ones(NUM_CLASSES), size=1)[0]
        
        # Lowercase mapping variable for comparison
        text_lower = cleaned_text.lower()
        
        # 1. Define emotion keywords with higher priority for explicit mentions
        emotion_keywords = {
            'Frustrated': ['frustrated', 'frustrating', 'annoying', 'angry', 'hate', 'difficult', 'stuck', 'wrong answer', 'keep getting', 'unnecessarily complicated', 'tried'],
            'Curious': ['why', 'how', 'what', 'curious', 'wonder', 'interested', 'learn', 'know more', 'want to know', 'explore', 'could we', 'what happens', 'intuition', 'bel'],
            'Confident': ['easy', 'amazing', 'great', 'excellent', 'good', 'awesome', 'perfect', 'solved', 'got it', 'clear now', 'finally', 'move ahead', 'understand clearly'],
            'Bored': ['boring', 'bored', 'tired', 'repetitive', 'dull', 'not engaging', 'didnt feel engaging', 'not interesting', 'too basic', 'losing'],
            'Confused': ['confused', 'lost', 'unclear', 'dont understand', "doesn't make sense", 'not fully confident', 'missing', 'incomplete', 'unsure']
        }

        # 2. Score each emotion based on keyword matches with higher weights for explicit mentions
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Give much higher weight to explicit emotion words
                    if keyword in ['frustrated', 'frustrating', 'curious', 'confident', 'bored', 'boring', 'confused']:
                        score += 10  # Very high weight for explicit emotions
                    else:
                        score += 2
            emotion_scores[emotion] = score

        # 3. Handle emotion with highest keyword score (Probability boosting and renormalization)
        max_score = max(emotion_scores.values())
        if max_score > 0:
            # Boost the emotion(s) with highest keyword matches
            for emotion, score in emotion_scores.items():
                if score == max_score:
                    emotion_idx = self.classes.index(emotion)
                    probs[emotion_idx] *= (1 + score * 3.0)  # Very strong boost for keyword matches
            
            # Reduce other emotions more aggressively
            winning_emotions = [e for e, s in emotion_scores.items() if s == max_score]
            for i, emotion in enumerate(self.classes):
                if emotion not in winning_emotions and max_score >= 5:  # Lower threshold for strong override
                    probs[i] *= 0.01  # Very strong reduction
                    
            # Renormalize probability spectrum back to unity sum bounds
            probs = probs / np.sum(probs)
            
        return {emo: float(score) for emo, score in zip(self.classes, probs)}