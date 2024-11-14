import ast
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import json
from collections import Counter
import nltk

# download punkt automatically, or manually
# pip install nltk==3.8.1
nltk.download('punkt')


class qa_dataset(data.Dataset):
    def __init__(self, data_dict):
        super(qa_dataset, self).__init__()
        self.data = data_dict
        self.begin_token = '<BEG>'
        self.end_token = '<END>'
        self.unknown_token = '<unk>'
        self.special_keys = [self.begin_token, self.end_token, self.unknown_token]

        sents = []
        for row in data_dict:
            sent = row['question']
            words = nltk.word_tokenize(sent)
            # TODO: if we did not make lower-case, how many words in the vocabulary
            # Compare uncased and cased vocabulary
            # read https://iq.opengenus.org/bert-cased-vs-bert-uncased/
            # explain which is better
            words = [word.lower() for word in words]  # make words lower case
            sents += words
            answer_choice = ast.literal_eval(row['answers'])
            keys = [k for k, n in answer_choice.items()]
            for word_key in keys:
                words = nltk.word_tokenize(word_key)
                words = [word.lower() for word in words]
                sents += words

        word_counter = Counter(sents)
        # print(word_counter)
        self.vocab_by_id = {}
        self.vocab_by_token = {}

        for ix, word_key in enumerate(word_counter):
            self.vocab_by_id[ix] = word_key
            self.vocab_by_token[word_key] = ix

        tmp_len = len(self.vocab_by_id)
        for ix, skey in enumerate(self.special_keys):
            self.vocab_by_id[tmp_len + ix] = skey
            self.vocab_by_token[skey] = tmp_len + ix

        print('Vocabulary has %d tokens' % (len(self.vocab_by_token),))
        self.pos_beg_token_id = self.vocab_by_token[self.begin_token]
        self.pos_end_token_id = self.vocab_by_token[self.end_token]
        self.pos_unknown_token_id = self.vocab_by_token[self.unknown_token]
        print(f'BEG {self.pos_beg_token_id}, END {self.pos_end_token_id}', )

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __get_str_tensor__(self, input_sent):
        # sentence convert to word index tensor
        words = nltk.word_tokenize(input_sent)
        words = [word.lower() for word in words]
        word_tensor = []
        # add [BEG] token to the head of question / answer string
        word_tensor.append(self.pos_beg_token_id)
        for ix, word in enumerate(words):
            if word in self.vocab_by_token:
                word_tensor.append(self.vocab_by_token[word])
            else:
                # unknown token
                word_tensor.append(self.pos_unknown_token_id)
        # append [END] token to the end of question / answer string
        word_tensor.append(self.pos_end_token_id)
        tensor = torch.tensor(word_tensor).long()
        return tensor

    def get_vocab_size(self):
        return len(self.vocab_by_id)

    def get_tensor(self, input_str):
        tensor = self.__get_str_tensor__(input_str)
        return tensor

    def get_word(self, index):
        return self.vocab_by_id[index]

    def get_beg_tensor(self):
        tensor = torch.tensor([self.pos_beg_token_id]).long()
        return tensor

    def __getitem__(self, index: int):
        query_str = self.data[index]['question'].strip()

        # Each query has several candidate answers with counts
        answer_choice = ast.literal_eval(self.data[index]['answers'])
        keys = [k for k, n in answer_choice.items()]

        # we prefer long answers, take their length as prob.
        answer_prob = torch.tensor([len(k.split()) for k, n in answer_choice.items()]).float()
        answer_id = torch.multinomial(answer_prob, 1)[0]

        answer_str = keys[answer_id].strip()

        # Encode question and answer string to tensor
        question_answer_tensors = [self.__get_str_tensor__(query_str),
                                   self.__get_str_tensor__(answer_str)]

        return question_answer_tensors
