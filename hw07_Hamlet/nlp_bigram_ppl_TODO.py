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
corpus_text = '''I play tennis. I like Chinese friends. I play tennis with Chinese friends. I have friends who like tennis.'''
token = nltk.word_tokenize(corpus_text)

###########################
# Generate Bi-Gram counter
###########################
unigrams = Counter([w[0] for w in list(ngrams(token, 1))])
bigrams = Counter(list(ngrams(token, 2)))


###########################
# generate query Bi-Gram
###########################
query_text_1 = "I play with Chinese friends"
query_text_2 = "Chinese friends who like tennis"
query_token = nltk.word_tokenize(query_text_1)
query_bigram = list(ngrams(query_token, 2))

# DO NOT MODIFY ABOVE


if debug:
    print(token)
    print(unigrams)
    print(bigrams)
    print(query_bigram)

###########################
# TODO: lookup each query bigram in each query_text
# compute Uni-Counter[bg[0]] /  Bi-Counter[(bg[0],bg[1])]
# convert to PPL and output
