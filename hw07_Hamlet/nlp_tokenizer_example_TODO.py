import os

import sentencepiece as spm
import numpy as np
from collections import defaultdict

import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

debug = False
###############################################
# generate Bi-Gram counter for training corpus
###############################################
corpus_text = '''I want to play tennis. I like Chinese friends. I play with Chinese classmates. I learn tennis with friends.'''
token = nltk.word_tokenize(corpus_text)

###########################
# Generate Bi-Gram counter
###########################
unigrams = Counter(list(ngrams(token, 1)))
bigrams = Counter(list(ngrams(token, 2)))



###########################
# generate query Bi-Gram
###########################
query_text = "I play tennis with Chinese friends."
query_token = nltk.word_tokenize(query_text)
query_bigram = list(ngrams(query_token, 2))

# DO NOT MODIFY ABOVE


if debug:
    print(token)
    print(unigrams)
    print(bigrams)
    print(query_bigram)


######################################################
# TODO: generate Uni-Gram word frequency table
# TODO: generate Bi-Gram frequency table

###########################
# TODO: lookup each query bigram in Bi-Gram frequency table and Uni-Gram frequency table
# then compute the log likelihood
