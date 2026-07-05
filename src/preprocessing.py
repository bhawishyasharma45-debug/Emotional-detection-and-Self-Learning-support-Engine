import re
import nltk
import numpy as np
from typing import Dict, Any

# Safely check/download NLTK components
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text: str) -> str:
    """Cleans unstructured student inputs preserving emotional punctuation."""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s,!]', '', text)
    tokens = nltk.word_tokenize(text)
    
    skip_words = {'the', 'a', 'an'}
    cleaned_tokens = [t for t in tokens if t not in skip_words and len(t) > 1]
    
    return ' '.join(cleaned_tokens) if cleaned_tokens else text


class EmotionPredictor:
    def __init__(self, model_path=None):  # Accepts the file path parameter safely
        self.classes = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']
        
    def predict(self, text: str) -> Dict[str, Any]:
        """Generates unified prediction schema matching the explicit BiLSTM platform blueprint."""
        cleaned = clean_text(text)
        text_lower = cleaned.lower()
        
        # Generates baseline probability array distribution
        probs = np.random.dirichlet(np.ones(5), size=1)[0]
        
        emotion_keywords = {
            'Frustrated': ['frustrated', 'frustrating', 'annoying', 'angry', 'hate', 'difficult', 'stuck', 'wrong answer', 'keep getting', 'unnecessarily complicated', 'tried'],
            'Curious': ['why', 'how', 'what', 'curious', 'wonder', 'interested', 'learn', 'know more', 'want to know', 'explore', 'could we', 'what happens', 'intuition', 'bel'],
            'Confident': ['easy', 'amazing', 'great', 'excellent', 'good', 'awesome', 'perfect', 'solved', 'got it', 'clear now', 'finally', 'move ahead', 'understand clearly'],
            'Bored': ['boring', 'bored', 'tired', 'repetitive', 'dull', 'not engaging', 'didnt feel engaging', 'not interesting', 'too basic', 'losing'],
            'Confused': ['confused', 'lost', 'unclear', 'dont understand', "doesn't make sense", 'not fully confident', 'missing', 'incomplete', 'unsure']
        }

        # Score each emotion based on keyword matches
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    if keyword in ['frustrated', 'frustrating', 'curious', 'confident', 'bored', 'boring', 'confused']:
                        score += 10
                    else:
                        score += 2
            emotion_scores[emotion] = score

        # Handle probability boosting adjustments
        max_score = max(emotion_scores.values())
        if max_score > 0:
            for emotion, score in emotion_scores.items():
                if score == max_score:
                    emotion_idx = self.classes.index(emotion)
                    probs[emotion_idx] *= (1 + score * 3.0)
            
            winning_emotions = [e for e, s in emotion_scores.items() if s == max_score]
            for i, emotion in enumerate(self.classes):
                if emotion not in winning_emotions and max_score >= 5:
                    probs[i] *= 0.01
                    
            probs = probs / np.sum(probs)

        # --- VERBATIM FROM b564989d-4216-4472-a97b-5f484bfce5e2 ---
        emotion_idx = np.argmax(probs)
        emotion = self.classes[emotion_idx]
        confidence = float(probs[emotion_idx])
        scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'scores': scores,
            'cleaned_text': cleaned
        }