from collections import defaultdict
from nltk.corpus import stopwords
import os
import re
import csv

class Lexicon_sentiment(object):
    # Set up the initial variables including world list and negation word
    def __init__(self, word_list = None, negations = None, intensifier = None):
        if word_list is None and negations is None and intensifier is None:
            base_dir = os.path.dirname(__file__)
            # financial sentiment word list - positive and negative words only
            word_list = os.path.join(base_dir, './word_list/LM_word_list.csv')
            # list of negation words
            negations = os.path.join(base_dir, './word_list/negation_words.csv')
            # intensifier list with its factor values
            intensifier = os.path.join(base_dir, './word_list/intensifier.csv')

        self.word_list = {}
        self.negations = set()
        self.intensifier = {}

        # load word list (can put multiple of them if needed)
        for wl_filename in self.__to_arg_list(word_list):
            self.load_word_list(wl_filename, self.word_list)
        # load negation words
        self.load_negations(negations)
        # load intensifier
        self.load_word_list(intensifier, self.intensifier)
        # a list of words to be skipped when checking prefixed negation words
        self.negation_skip = {'a', 'an', 'so', 'too'}

    # transform the input argument into list -> to be used in init function
    @staticmethod
    def __to_arg_list(obj):
        if obj is not None:
            if not isinstance(obj, list):
                obj = [obj]
        else:
            obj = []
        return obj

    def load_negations(self, filename):
        with open(filename, 'r') as f:
            has_header = csv.Sniffer().has_header(f.read(1024))
            f.seek(0) # rewind
            reader = csv.reader(f)
            if has_header:
                next(reader) # skip header row

            negations = set([row[0] for row in reader])
        self.negations = negations

    # helper function to load word list with its numerical values as well
    # numerical value can be intensifier factor, sentiment value, etc
    def load_word_list(self, filename, destination):
        with open(filename, 'r') as f:
            has_header = csv.Sniffer().has_header(f.read(1024))
            f.seek(0) # rewind
            reader = csv.reader(f)
            next(reader)  # skip header row

            word_list = {row[0]: float(row[1]) for row in reader}
        destination.update(word_list)

    # function to check if the current word is prefixed by a negations  word or not
    # it is to be used to flip the Sentiment if needed
    # Return True if index != 0 and tokens[i-1] is in self.negations else False
    def is_prefixed_by_negation(self, token_index, tokens):
        prev_index = token_index - 1
        if tokens[prev_index] in self.negation_skip or tokens[prev_index] in self.intensifier:
            prev_index -= 1 # if the previous token is skip word, shift the index by -1

        is_prefixed_by_negation = False

        if token_index > 0 and prev_index >= 0 and tokens[prev_index] in self.negations:
            is_prefixed_by_negation = True

        return is_prefixed_by_negation

    # function to adjust the sentiment value with negation word
    def negate_sentiment(self, sentiment):
        flipped_sentiment = sentiment * -1

        return 0.5 * flipped_sentiment

    # function to return the impact of intensifier prior to the current word
    def intensified_by(self, token_index, tokens):
        prev_word = tokens[token_index -1]
        factor = 1 # default output

        if prev_word in self.intensifier:
            factor = self.intensifier[prev_word]

        return factor

    # main function to allocate sentiment value to the text
    def assign_sentiment(self, text):
        # clean up the text first and change it into uppercase letters
        text_clean = re.sub(r'[^\w ]', ' ', text.upper())
        useful_tokens = text_clean.split()
        # useful_tokens = [word for word in useful_tokens if word not in stopwords.words('english')]

        scores = defaultdict(float) # sentiment scores to be stored
        # words = defaultdict(list) # positive and negative words detected to be stored
        comparative = 0

        for i, token in enumerate(useful_tokens):
            is_prefixed_by_negation = self.is_prefixed_by_negation(i, useful_tokens)
            intensified_by = self.intensified_by(i, useful_tokens)

            # if the word is in the word list and not prefixed by negations word:
            if token in self.word_list:
                if not is_prefixed_by_negation:
                    score = self.word_list[token] * intensified_by
                    score_type = 'negative' if score < 0 else 'positive'
                    scores[score_type] += score
                    # words[score_type].append(token)
                else:
                    score = self.word_list[token] * intensified_by
                    score = self.negate_sentiment(score)
                    score_type = 'negative' if score < 0 else 'positive'
                    scores[score_type] += score

        if len(useful_tokens) > 0:
            # calculate comparative score / result to output
            # TODO: apply better normalisation method
            comparative = (scores['positive'] + scores['negative']) / len(useful_tokens)

        return comparative

def main():
    sentiment = Lexicon_sentiment()

    article = "company is doing not really abnormal compared to last year"

    result = sentiment.assign_sentiment(article)
    print(result)

if __name__ == '__main__':
    main()
