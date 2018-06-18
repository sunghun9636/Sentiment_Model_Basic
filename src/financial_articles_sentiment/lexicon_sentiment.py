from collections import defaultdict
import os
import re
import csv

class Sentiment(object):
    # Set up the initial variables including world list and negation word
    def __init__(self, word_list = None, negation = None):
        if word_list is None and negation is None:
            base_dir = os.path.dirname(__file__)
            word_list = os.path.join(base_dir, './word_list/LM_word_list.csv')
            negation = os.path.join(base_dir, './word_list/negation_words.csv')

        self.word_list = {}
        self.negations = set()

        for wl_filename in self.__to_arg_list(word_list):
            self.load_word_list(wl_filename)
        for negations_filename in self.__to_arg_list(negation):
            self.load_negations(negations_filename)

        self.__negation_skip = {'a', 'an', 'so', 'too'}

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
            reader = csv.DictReader(f)
            negations = set([row['token'] for row in reader])
        self.negations = negations

    def load_word_list(self, filename):
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            word_list = {row['word']: float(row['score']) for row in reader}
        self.word_list.update(word_list)

    # function to check if the current word is prefixed by a negation word or not
    # it is to be used to flip the Sentiment if needed
    def is_prefixed_by_negation(self, token_index, tokens):
        # Return True if index != 0 and tokens[i-1] is in self.negations else False

        prev_index = token_index - 1
        if tokens[prev_index] in self.__negation_skip:
            prev_index -= 1 # if the previous token is skip word, shift the index by -1

        is_prefixed_by_negation = False

        if token_index > 0 and prev_index >= 0 and tokens[prev_index] in self.negations:
            is_prefixed_by_negation = True

        return is_prefixed_by_negation

    
