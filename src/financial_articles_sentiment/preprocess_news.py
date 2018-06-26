import nltk
import re, string, unicodedata
import contractions
import inflect
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# Noise Removal
def replace_contractions(text):
    # e.g. "I'm doing this" -> "I am doing this"
    return contractions.fix(text)

# Tokenization

# Normalization
def remove_non_ascii(words):
    clean_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        clean_words.append(new_word)

    return clean_words

def to_uppercase(words):
    new_words = []
    for word in words:
        new_word = word.upper()
        new_words.append(new_word)

    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

    
