import re
import nltk
from nltk.corpus import stopwords

# Safely download NLTK data packets if they aren't already present in your environment
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text: str) -> str:
    """
    Cleans unstructured student inputs:
    1. Converts everything to lowercase.
    2. Strips out URLs and hypertext pathways.
    3. Removes numbers and punctuation symbols.
    4. Filters out standard English stopwords and lonely single letters.
    """
    if not text:
        return ""
    
    # 1. Lowercase folding
    text = str(text).lower()
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    
    # 3. Keep only alphabetical letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 4. Tokenize and filter out noise
    tokens = nltk.word_tokenize(text)
    english_stopwords = set(stopwords.words('english'))
    
    cleaned_tokens = [t for t in tokens if t not in english_stopwords and len(t) > 1]
    
    return " ".join(cleaned_tokens)