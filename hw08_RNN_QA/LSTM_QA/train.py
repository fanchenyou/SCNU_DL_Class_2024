import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time
import csv
from model import *
from generate import generate
from torch.utils import data
from torch.utils.data import DataLoader
from qa_dataset import qa_dataset
import pickle

def save(model):
    save_filename = "saves/rnn.pt"
    if not os.path.isdir('saves'):
        os.mkdir('saves')
    # save entire model along with its parameters
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


def main(args):
    # if you have a GPU, turn this on, or pass --gpu in command line
    use_cuda = args.cuda and torch.cuda.is_available()
    # if you have a MAC M3/4 chip, turn this on
    use_mps = False  # args.mps or torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print('The device is', device)

    # read in QA dataset
    data_dict = []
    with open('train_proto_qa.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_dict.append(row)
    # visualize some data, also you can vis in csv file
    # print(len(data_dict))
    # print(data_dict[:20])

    # for this example, we just use train data
    # we save the vocabulary and data into a pickle file, which can be used for decoding in generate.py
    train_set = qa_dataset(data_dict)
    vocab_size = train_set.get_vocab_size()
    with open('saves/vocab.pkl','wb') as f:
        pickle.dump(train_set, f)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True)


    def train(model, optimizer, epoch):
        # accumulate gradient over batch
        optimizer.zero_grad()
        loss_avg = 0
        acc_train_iter = 0
        acc_backward_num = 0
        for batch_idx, (question_tensor, answer_tensor) in enumerate(train_loader):
            if batch_idx % 1000 == 0:
                print('[Epoch %d] Iter %d' % (epoch, batch_idx))
            question_tensor, answer_tensor = question_tensor.to(device), answer_tensor.to(device)
            # print(question_tensor.size(), answer_tensor.size())
            hidden = model.init_hidden(1, device)

            ##################################
            ########## TODO-Explain ##########
            ##################################
            # TODO: Explain why give in answer_tensor[:, :-1] in forward(), hint: the last token is [END], check qa_dataset
            output = model(question_tensor, answer_tensor[:, :-1], hidden)
            # TODO: Explain why use answer_tensor[:, 1:] as training target, hint: the first token is [BEG]
            loss = criterion(output.squeeze(0), answer_tensor[:, 1:].squeeze(0)) / args.batch_size
            # hint: review teacher forcing in lecture note

            # TODO-Explain: understand why we use accumulative gradient method
            # read https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20?u=alband
            # but here we need accumulative gradient for different purpose
            # each question and answer has different length
            # it's hard to put them in a batch (an alternative way is to using pad_sequence, but complex)
            # so here we just use batch_size=1, and accumulate args.batch_size gradients
            loss_avg += loss.item()
            loss.backward()

            # TODO-Explain (continue): here, when we accumulate a batch_size of gradients
            # we perform just one step of optimization.
            # hint: 1) forwarding 32 times of batch_size 1 and performing backward once;
            # 2) forwarding a batch of 32 sequences all together and backward once.  Is 1) and 2) equivalent?
            acc_train_iter += 1
            if acc_train_iter % args.batch_size == 0:
                acc_backward_num += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

        if acc_backward_num == 0:
            return 0
        return loss_avg / acc_backward_num

    # initialize models
    model = QA_RNN(
        vocab_size,
        args.hidden_size,
        model=args.model,
        n_layers=args.n_layers,
    ).to(device)

    # if you want to resume training, uncomment this
    if False:
        if os.path.isfile(args.model_path):
            model = torch.load(args.model_path, map_location='cpu').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # start = time.time()
    loss_avg = 0

    debug = False
    if debug:
        args.n_epochs = 100
        args.print_every = 5

    history_responses = []
    try:
        print("Training for %d epochs..." % args.n_epochs)
        for epoch in tqdm(range(1, args.n_epochs + 1)):
            loss = train(model, optimizer, epoch)
            loss_avg += loss

            if epoch % args.print_every == 0:
                # you can change the questions which print out after each epoch
                # check if some interesting answers can be generated
                query_strs = ['Name a good sport.','Name a famous person','Tell me something funny']
                responses = []
                for query_str in query_strs:
                    response = generate(model, train_set , query_str, 100, device=device)
                    print('Q: %s' % (query_str,))
                    print('A: %s' % (response,))
                    responses.append(response)
                history_responses.append(responses)
                save(model)

            if epoch % 10 == 0:
                print('history response', history_responses)

        print("Saving...")
        save(model)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(model)


if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filename', type=str)
    argparser.add_argument('--model', type=str, default="gru", choices=['gru','lstm'])
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--print_every', type=int, default=1)
    argparser.add_argument('--hidden_size', type=int, default=128)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--learning_rate', type=float, default=0.006)
    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('-m', '--model_path', type=str, default='saves/rnn.pt')

    args = argparser.parse_args()

    main(args)
