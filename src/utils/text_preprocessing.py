"""
Text preprocessing and tokenization utilities.
"""

import re
import string
from typing import List, Dict, Any
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import AutoTokenizer
import pandas as pd

class TextPreprocessor:
    """Class for text preprocessing operations."""
    
    def __init__(self):
        self.nlp = None
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def load_spacy_model(self, model_name: str = "en_core_web_sm"):
        """Load spaCy model for advanced preprocessing."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}")
    
    def basic_clean(self, text: str) -> str:
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for tweets)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_punctuation(self, text: str, keep_sentence_endings: bool = True) -> str:
        """Remove punctuation from text."""
        if keep_sentence_endings:
            # Keep sentence ending punctuation
            text = re.sub(r'[^\w\s.!?]', '', text)
        else:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_stopwords(self, text: str, language: str = 'english') -> str:
        """Remove stopwords from text."""
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words(language))
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        return ' '.join(filtered_words)
    
    def lemmatize(self, text: str) -> str:
        """Lemmatize text using spaCy."""
        if self.nlp is None:
            self.load_spacy_model()
        
        if self.nlp is None:
            return text
        
        doc = self.nlp(text)
        lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        
        return ' '.join(lemmatized)
    
    def preprocess_text(self, text: str, steps: List[str] = None) -> str:
        """
        Apply multiple preprocessing steps.
        
        Args:
            text: Input text
            steps: List of preprocessing steps to apply
                   Options: ['basic_clean', 'lowercase', 'remove_punctuation', 
                           'remove_stopwords', 'lemmatize']
        """
        if steps is None:
            steps = ['basic_clean', 'lowercase', 'remove_punctuation']
        
        processed_text = text
        
        for step in steps:
            if step == 'basic_clean':
                processed_text = self.basic_clean(processed_text)
            elif step == 'lowercase':
                processed_text = self.to_lowercase(processed_text)
            elif step == 'remove_punctuation':
                processed_text = self.remove_punctuation(processed_text)
            elif step == 'remove_stopwords':
                processed_text = self.remove_stopwords(processed_text)
            elif step == 'lemmatize':
                processed_text = self.lemmatize(processed_text)
        
        return processed_text

class TokenizerManager:
    """Manage different tokenization methods."""
    
    def __init__(self):
        self.tokenizers = {}
    
    def create_tfidf_vectorizer(self, max_features: int = 5000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
        """Create TF-IDF vectorizer."""
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        return vectorizer
    
    def create_count_vectorizer(self, max_features: int = 5000, ngram_range: tuple = (1, 2)) -> CountVectorizer:
        """Create Count vectorizer."""
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        return vectorizer
    
    def create_transformer_tokenizer(self, model_name: str = "bert-base-uncased") -> AutoTokenizer:
        """Create transformer tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    
    def tokenize_for_transformer(self, texts: List[str], tokenizer: AutoTokenizer, 
                                max_length: int = 512) -> Dict[str, Any]:
        """Tokenize texts for transformer models."""
        encoding = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding

def preprocess_dataset(df: pd.DataFrame, text_column: str, 
                      preprocessing_steps: List[str] = None) -> pd.DataFrame:
    """Preprocess entire dataset."""
    preprocessor = TextPreprocessor()
    
    df = df.copy()
    df[f'{text_column}_processed'] = df[text_column].apply(
        lambda x: preprocessor.preprocess_text(x, preprocessing_steps)
    )
    
    return df
