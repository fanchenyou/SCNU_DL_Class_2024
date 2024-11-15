import torch
import torch.nn as nn

class QA_RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, model="lstm", n_layers=1):
        super(QA_RNN, self).__init__()
        self.model = model.lower()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.n_layers = n_layers

        # TODO-Explain, why we need to separate encoder and decoder
        # Find the encoder-decoder architecture in lecture note
        # TODO: Draw the task computational graph of our word-based question-answering
        self.embed = nn.Embedding(vocab_size, hidden_size)
        if self.model == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
            self.decoder = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
        elif self.model == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
            self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_question, target_answer, hidden):
        batch_size = input_question.size(0)
        assert batch_size == 1
        question_embed = self.embed(input_question)  # [B=1, Question_len, hidden_size]
        answer_embed = self.embed(target_answer)  # [B=1, Answer_len, hidden_size]
        output_enc, hidden_enc = self.encoder(question_embed, hidden)
        output_dec, hidden_dec = self.decoder(answer_embed, hidden_enc)

        pred_word_dec = self.fc(output_dec)
        return pred_word_dec

    def generate(self, input_question, answer_init_token, pos_end_token, max_predict_len, device):

        hidden = self.init_hidden(1, device)
        question_embed = self.embed(input_question)  # [B=1, Question_len, hidden_size]
        question_embed = question_embed.unsqueeze(0)

        # In inference, the input is [BEG] only, so its size is [1,1,D]
        answer_init_embed = self.embed(answer_init_token)
        answer_init_embed = answer_init_embed.unsqueeze(0)
        ########################################
        #### Stage-1: Encoding Query String  ###
        ########################################
        # print(question_embed.device, hidden[0].device)
        output_enc, hidden_enc = self.encoder(question_embed, hidden)

        #######################################
        #### Stage-2 decoding as response  ####
        #######################################
        # TODO: Explain stage-2 as answer generation
        # draw a pipeline as in lecture note about inference-stage
        # draw encoder and decoder
        # note that we pass encoder hidden state to decoder as init hidden state
        # note that we should start from BEG token as first input (x0) to decoder
        token_cur = answer_init_embed
        hidden_cur = hidden_enc
        pred_tokens = []
        for p in range(max_predict_len):
            feat, hidden_cur = self.decoder(token_cur, hidden_cur)
            pred_token = self.fc(feat)

            # The predicted word is from argmax the output probability of each word in vocab
            top_i = torch.argmax(pred_token, dim=2)
            token_cur = self.embed(top_i)

            word_index = top_i.item()
            pred_tokens.append(word_index)
            if word_index == pos_end_token:
                break

        return pred_tokens

    def init_hidden(self, batch_size, device):
        if self.model == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))
        # GRU
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
