import torch
import argparse
from qa_dataset import qa_dataset
import pickle

def generate(model, train_dataset, query_str='A', max_predict_len=100, device=None):

    # edge case, put a default query
    if len(query_str) == 0:
        query_str = 'A'

    # encode query string to word tensor, you can print it out
    query_input = train_dataset.get_tensor(query_str).to(device)

    # TODO: Explain, why we need pass [BEG] token into generate
    query_answer_init_token = train_dataset.get_beg_tensor().to(device)
    # TODO: Explain function of encoding stage
    predicted = model.generate(query_input, query_answer_init_token,
                               train_dataset.pos_end_token_id, max_predict_len, device=device)

    # if last token is [END], remove it and not print
    if predicted[-1] == train_dataset.pos_end_token_id:
        predicted = predicted[:-1]

    # convert token index back into word string
    word_list = [train_dataset.get_word(ind) for ind in predicted]
    answer_str = ' '.join(word_list)

    return answer_str


# Run as standalone script
if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model_path', type=str, default='saves/rnn.pt')
    argparser.add_argument('-v', '--vocab_path', type=str, default='saves/vocab.pkl')
    argparser.add_argument('-q', '--query_str', type=str, required=True)
    argparser.add_argument('-l', '--predict_len', type=int, default=300)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    # if you have a GPU, turn this on
    use_cuda = args.cuda and torch.cuda.is_available()

    # if you have a MAC M3/4 chip, turn this on
    use_mps = False  # args.mps or torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # print('The device is', device)
    with open(args.vocab_path,'rb') as f:
        train_set = pickle.load(f)


    model = torch.load(args.model_path, map_location='cpu').to(device)
    response = generate(model, train_set, args.query_str, args.predict_len, device)
    print('Q: %s' % (args.query_str,))
    print('A: %s' % (response,))
