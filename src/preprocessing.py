import re
import nltk

# Safely download NLTK data packets if they aren't already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text: str) -> str:
    """
    Cleans unstructured student inputs matching Epic 3 workspace requirements.
    """
    if not text:
        return ""
    
    # 1. Lowercase folding
    text = str(text).lower()
    
    # 2. Keep punctuation that indicates emotion (letters, spaces, commas, exclamations)
    text = re.sub(r'[^a-zA-Z\s,!]', '', text)
    
    # 3. Tokenize text layout
    tokens = nltk.word_tokenize(text)
    
    # 4. Keep ALL meaningful words, remove only basic articles
    skip_words = {'the', 'a', 'an'}
    cleaned_tokens = [t for t in tokens if t not in skip_words and len(t) > 1]
    
    return ' '.join(cleaned_tokens) if cleaned_tokens else text