import nltk
import re, string, unicodedata
import contractions
import inflect
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

## Extract meaningful paragraphs

# return if the paragraph contains over n number of numerical figures
# to be used to filter paragraphs stating factual numbers only
def is_numerical(text):
    count = 0 # counter for numerical figures / expressions
    words = nltk.word_tokenize(text)
    for word in words:
        if word.isdigit():
            count += 1

    return count > 5 # TODO: change the 'n' to appropriate number

## Noise Removal

def replace_contractions(text):
    # e.g. "I'm doing this" -> "I am doing this"
    return contractions.fix(text)

## Normalization

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


## Main function to preprocess the financial news articles
def preprocess(text):
    # 1. extract meaningful paragraphs
    paragraphs = text.split('\n')
    # a. remove paragraphs with many numerical figures
    non_numerical_paragraphs = []
    for paragraph in paragraphs:
        if not is_numerical(paragraph):
            non_numerical_paragraphs.append(paragraph)
    # b. remove paragraph that doesn't mention the company at all
    # c. TODO: further remove unnecessary paragraphs
    clean_text = []
    for paragraph in non_numerical_paragraphs:
        clean_text.append(' ')
        clean_text.append(paragraph)
    clean_text = ''.join(clean_text)

    # 2. Noise Removal
    clean_text = replace_contractions(clean_text)

    # 3. Tokenization
    words = nltk.word_tokenize(clean_text)

    # 4. Normalization

if __name__ == '__main__':
    text = "Company A has increased its stock price by 5%. I'm testing.\n I am testing"
    paragraphs = text.split('\n')
    words = nltk.word_tokenize(paragraphs[1])
    for word in words:
        print(word)
