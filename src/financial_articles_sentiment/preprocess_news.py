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
    maximum_numericals = 10 # TODO: change the 'n' to appropriate number
    words = nltk.word_tokenize(text)
    for word in words:
        if word.isdigit():
            count += 1

    return count > maximum_numericals

# return if the paragraph contain the company name or not
def has_company_name(text, name):
    words = nltk.word_tokenize(text)

    if name in words:
        return True
    else:
        return False

## Noise Removal

# e.g. "I'm doing this" -> "I am doing this"
def replace_contractions(text):
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

def remove_stopwords(words):
    new_words = [word for word in words if word not in stopwords.words('english')]
    return new_words

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []

    for word in words:
        lemma = lemmatizer.lemmatize(word, pos = 'v')
        lemmas.append(lemma)
    return lemmas

## Main function to preprocess the financial news articles
def preprocess(text, company_name):
    # 1. extract meaningful paragraphs
    paragraphs = text.split('\n')
    # a. remove paragraphs with many numerical figures
    non_numerical_paragraphs = []
    for paragraph in paragraphs:
        if not is_numerical(paragraph):
            non_numerical_paragraphs.append(paragraph)
    # b. remove paragraph that doesn't mention the company at all
    # b. remove the whole article if it is a clickbait
    useful_paragraphs = []
    for paragraph in non_numerical_paragraphs:
        if has_company_name(paragraph, company_name):
            useful_paragraphs.append(paragraph)
    # c. TODO: further remove unnecessary paragraphs

    clean_text = []
    for paragraph in useful_paragraphs:
        clean_text.append(' ')
        clean_text.append(paragraph)
    clean_text = ''.join(clean_text)

    # 2. Noise Removal
    clean_text = replace_contractions(clean_text)

    # 3. Tokenization
    words = nltk.word_tokenize(clean_text)

    # 4. Normalization
    words = remove_non_ascii(words)
    words = to_uppercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)

    return words

if __name__ == '__main__':
    text = "Facebook is doing really poorly this year.\n\n Also, its stock price has dropped by 5%, ... Facebook"
    company_name = "Facebook"
    words = preprocess(text, company_name)
    for word in words:
        print(word)
