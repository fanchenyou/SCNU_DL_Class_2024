import os
import sentencepiece as spm
import numpy as np
from collections import defaultdict
import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

# YOU CAN TURN ON/OFF debug FLAG
debug = True

#####################################
########## Read Shakespeare #########
#####################################
corpus = []
for file_name in os.listdir('shakespeare-db'):
    # print(file_name)
    with open(os.path.join('shakespeare-db', file_name)) as file:
        corpus += [line.strip() for line in file]

if debug:
    print(corpus[:10])

the_question = "To be, or not to be: that is the question:"
the_question_tokens = nltk.word_tokenize(the_question)

if debug:
    ###############################################
    ## TODO: you please answer Hamlet's question ##
    ## This is open question.
    ###############################################
    # we can locate this sentence in Act 3, Scene 1
    # go find this famous phrase  in shakespeare/Hamlet.txt
    # print this sentence Context, translate to Chinese
    id_q = corpus.index(the_question)
    print(corpus[id_q])
    print(the_question_tokens)

################################################
## tokenize each sentence into word sequences ##
################################################
tokens = []
for ix, line in enumerate(corpus):
    tokens += nltk.word_tokenize(line)
    if debug and ix % 1000 == 0:
        print(nltk.word_tokenize(line))

# DO NOT MODIFY ABOVE
# YOU CAN TURN ON/OFF debug FLAG

# TODO, compute the log likelihood of the_question, based on Bi-Gram
# Refer to Q1
