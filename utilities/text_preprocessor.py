import re
import unicodedata
import emoji
import json
from typing import Optional, Union, List
from nltk.corpus import stopwords
import spacy

# Constantes
DEFAULT_CONFIG = {
    "remove_emojis": False,
    "replace_emojis": False,
    "describe_emojis": False,
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
    "language": "english",
    "custom_stopwords": [],
    "to_lower": False
}

# Recursos NLP
class NLPResources:
    _nlp = None
    _stop_words = None

    @staticmethod
    def get_nlp(model: str = "en_core_web_sm"):
        if NLPResources._nlp is None:
            try:
                NLPResources._nlp = spacy.load(model)
            except ImportError:
                raise ImportError(f"SpaCy model not found. Run: python -m spacy download {model}")
        return NLPResources._nlp

    @staticmethod
    def get_stopwords(language: str = "english", custom_stopwords: Optional[list] = None):
        if NLPResources._stop_words is None:
            try:
                NLPResources._stop_words = set(stopwords.words(language))
            except LookupError:
                import nltk
                nltk.download("stopwords")
                NLPResources._stop_words = set(stopwords.words(language))
            if custom_stopwords:
                NLPResources._stop_words.update(custom_stopwords)
        return NLPResources._stop_words

# Validación de configuración
def validate_config(config: dict) -> dict:
    validated_config = config.copy()
    bool_keys = [
        "remove_emojis", "replace_emojis", "describe_emojis", "remove_urls", "replace_urls",
        "remove_mentions_and_hashtags", "replace_mentions", "replace_hashtags",
        "remove_numbers", "replace_numbers", "remove_special_characters",
        "replace_special_characters", "remove_accents", "remove_stopwords", "lemmatize", "to_lower"
    ]
    str_keys = [
        "emoji_replacement", "url_replacement", "mention_replacement",
        "hashtag_replacement", "number_replacement", "special_character_replacement", "language"
    ]

    for key in bool_keys:
        if key in validated_config and not isinstance(validated_config[key], bool):
            print(f"Warning: Invalid value for {key}. Expected boolean, got {type(validated_config[key])}. Using default.")
            validated_config[key] = DEFAULT_CONFIG[key]

    for key in str_keys:
        if key in validated_config and not isinstance(validated_config[key], str):
            print(f"Warning: Invalid value for {key}. Expected string, got {type(validated_config[key])}. Using default.")
            validated_config[key] = DEFAULT_CONFIG[key]

    if validated_config.get("language") not in stopwords.fileids():
        print(f"Warning: Language {validated_config['language']} not supported. Falling back to 'english'.")
        validated_config["language"] = "english"

    return validated_config

# Funciones de procesamiento
def process_emojis(text: str, remove: bool = False, replace: bool = False, 
                  describe: bool = False, replacement: str = "EMOJI") -> str:
    if remove:
        return emoji.replace_emoji(text, '')
    if replace:
        return emoji.replace_emoji(text, replacement)
    if describe:
        return emoji.replace_emoji(text, lambda m: emoji.demojize(m.group()))
    return text

def process_urls(text: str, remove: bool = False, replace: bool = False, replacement: str = "URL") -> str:
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    if remove:
        return re.sub(url_pattern, '', text)
    if replace:
        return re.sub(url_pattern, replacement, text)
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

def normalize_case(text: str, to_lower: bool = False) -> str:
    return text.lower() if to_lower else text

def remove_stopwords(text: str, language: str = "english", custom_stopwords: Optional[list] = None) -> str:
    stop_words = NLPResources.get_stopwords(language, custom_stopwords)
    words = text.split()
    return ' '.join(word for word in words if word.lower() not in stop_words)

def lemmatize_text(text: str) -> str:
    nlp = NLPResources.get_nlp()
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_stop)

def tokenize_text(text: str, lemmatize: bool = False) -> list:
    nlp = NLPResources.get_nlp()
    doc = nlp(text)
    if lemmatize:
        return [token.lemma_ for token in doc if not token.is_stop]
    return [token.text for token in doc if not token.is_stop]

def load_config(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            user_config = json.load(f)
            return validate_config({**DEFAULT_CONFIG, **user_config})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Config loading error: {str(e)}. Using default configuration.")
        return DEFAULT_CONFIG

def clean_text(text: str, config_path: Optional[str] = None, return_tokens: bool = False) -> Optional[Union[str, list]]:
    """
    Limpia un texto aplicando varias transformaciones según la configuración proporcionada.

    Args:
        text (str): El texto a limpiar.
        config_path (Optional[str]): Ruta al archivo JSON con la configuración.
        return_tokens (bool): Si es True, devuelve una lista de tokens en lugar de un string.

    Returns:
        Optional[Union[str, list]]: El texto limpio o una lista de tokens, o None si hay un error.
    """
    if not isinstance(text, str) or not text:
        return None

    config = load_config(config_path) if config_path else DEFAULT_CONFIG
    config = validate_config(config)

    try:
        text = process_emojis(text, config.get("remove_emojis"), config.get("replace_emojis"),
                            config.get("describe_emojis"), config.get("emoji_replacement"))
        text = process_urls(text, config.get("remove_urls"), config.get("replace_urls"),
                          config.get("url_replacement"))
        text = process_mentions_and_hashtags(text, config.get("remove_mentions_and_hashtags"),
                                           config.get("replace_mentions"), config.get("replace_hashtags"),
                                           config.get("mention_replacement"),
                                           config.get("hashtag_replacement"))
        text = process_numbers(text, config.get("remove_numbers"), config.get("replace_numbers"),
                             config.get("number_replacement"))
        text = process_special_characters(text, config.get("remove_special_characters"),
                                        config.get("replace_special_characters"),
                                        config.get("special_character_replacement"))
        
        if config.get("remove_accents"):
            text = remove_accents(text)
        
        text = normalize_whitespace(text)
        
        if config.get("to_lower"):
            text = normalize_case(text, to_lower=True)
        
        if config.get("remove_stopwords"):
            text = remove_stopwords(text, config.get("language"), config.get("custom_stopwords"))
            
        if config.get("lemmatize"):
            text = lemmatize_text(text)

        if return_tokens:
            return tokenize_text(text, config.get("lemmatize"))
        
        return text if text else None

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None