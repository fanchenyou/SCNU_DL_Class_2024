import os
import nltk
import numpy as np
from transformers import AutoTokenizer
from nltk.util import ngrams
from collections import Counter

# TODO
# download tokenizer from https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main
# tokenizer.json
# special_tokens_map.json
# tokenizer_config.json
# put them in ./model
tokenizer = AutoTokenizer.from_pretrained('./model')

#####################################
########## Do not modify ###########
#####################################
corpus = []
for file_name in os.listdir('shakespeare-db'):
    #print(file_name)
    with open(os.path.join('shakespeare-db',file_name)) as file:
        corpus += [line.strip() for line in file]
# tokenize sentences into tokens
# do not modify
sent_all = []
for text in corpus:
    words = tokenizer.tokenize(text, add_special_tokens=True)
    sent_all += words


###########################
# Generate Bi-Gram counter
###########################
unigrams = Counter([w[0] for w in list(ngrams(sent_all, 1))])
bigrams = Counter(list(ngrams(sent_all, 2)))
# tokenize THE QUESTION
the_question = "To be, or not to be: that is the question"
the_question_tokens = tokenizer.tokenize(the_question)

# TODO: compute PPL here
