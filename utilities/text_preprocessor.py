import re
import unicodedata
import emoji
import json
from typing import Optional
from nltk.corpus import stopwords
import spacy

# Initialize resources
def initialize_nlp(model: str = "en_core_web_sm"):
    try:
        return spacy.load(model)
    except:
        raise ImportError(f"SpaCy model not found. Run: python -m spacy download {model}")

def initialize_stopwords(language: str = "english"):
    try:
        return set(stopwords.words(language))
    except LookupError:
        import nltk
        nltk.download("stopwords")
        return set(stopwords.words(language))

nlp = initialize_nlp()
stop_words = initialize_stopwords()

# Core text processing functions
def process_emojis(text: str, remove: bool = False, replace: bool = False, replacement: str = "EMOJI") -> str:
    if remove:
        return emoji.replace_emoji(text, '')
    if replace:
        return emoji.replace_emoji(text, replacement)
    return text

def process_urls(text: str, remove: bool = False, replace: bool = False, replacement: str = "URL") -> str:
    if remove:
        return re.sub(r'http\S+|www\S+|https\S+', '', text)
    if replace:
        return re.sub(r'http\S+|www\S+|https\S+', replacement, text)
    return text

def process_mentions_and_hashtags(text: str, remove: bool = False, replace_mentions: bool = False, 
                                replace_hashtags: bool = False, mention_replacement: str = "MENTION", 
                                hashtag_replacement: str = "HASHTAG") -> str:
    if remove:
        text = re.sub(r'@\w+|#\w+', '', text)
    elif replace_mentions:
        text = re.sub(r'@\w+', mention_replacement, text)
    elif replace_hashtags:
        text = re.sub(r'#\w+', hashtag_replacement, text)
    return text

def process_numbers(text: str, remove: bool = False, replace: bool = False, replacement: str = "NUMBER") -> str:
    if remove:
        return re.sub(r'\d+', '', text)
    if replace:
        return re.sub(r'\d+', replacement, text)
    return text

def process_special_characters(text: str, remove: bool = False, replace: bool = False, replacement: str = " ") -> str:
    if remove:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    if replace:
        return re.sub(r'[^a-zA-Z0-9\s]', replacement, text)
    return text

def remove_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) 
                  if unicodedata.category(c) != 'Mn')

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text: str) -> str:
    words = text.split()
    return ' '.join(word for word in words if word.lower() not in stop_words)

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_stop)

# Configuration handling
def load_config(file_path: str) -> dict:
    default_config = {
        "remove_emojis": False,
        "replace_emojis": False,
        "emoji_replacement": "EMOJI",
        "remove_urls": False,
        "replace_urls": False,
        "url_replacement": "URL",
        "remove_mentions_and_hashtags": False,
        "replace_mentions": False,
        "replace_hashtags": False,
        "mention_replacement": "MENTION",
        "hashtag_replacement": "HASHTAG",
        "remove_numbers": False,
        "replace_numbers": False,
        "number_replacement": "NUMBER",
        "remove_special_characters": False,
        "replace_special_characters": False,
        "special_character_replacement": " ",
        "remove_accents": False,
        "remove_stopwords": False,
        "lemmatize": False,
        "language": "english"
    }
    
    try:
        with open(file_path, 'r') as f:
            user_config = json.load(f)
            return {**default_config, **user_config}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Config loading error: {str(e)}")
        return default_config

def clean_text(text: str, config_path: Optional[str] = None) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None

    config = load_config(config_path) if config_path else {}
    
    text = process_emojis(text, config.get("remove_emojis"), config.get("replace_emojis"),
                         config.get("emoji_replacement", "EMOJI"))
    text = process_urls(text, config.get("remove_urls"), config.get("replace_urls"),
                       config.get("url_replacement", "URL"))
    text = process_mentions_and_hashtags(text, config.get("remove_mentions_and_hashtags"),
                                       config.get("replace_mentions"), config.get("replace_hashtags"),
                                       config.get("mention_replacement", "MENTION"),
                                       config.get("hashtag_replacement", "HASHTAG"))
    text = process_numbers(text, config.get("remove_numbers"), config.get("replace_numbers"),
                         config.get("number_replacement", "NUMBER"))
    text = process_special_characters(text, config.get("remove_special_characters"),
                                    config.get("replace_special_characters"),
                                    config.get("special_character_replacement", " "))
    
    if config.get("remove_accents"):
        text = remove_accents(text)
        
    text = normalize_whitespace(text)
    
    if config.get("remove_stopwords"):
        text = remove_stopwords(text)
        
    if config.get("lemmatize"):
        text = lemmatize_text(text)

    return text if text else None