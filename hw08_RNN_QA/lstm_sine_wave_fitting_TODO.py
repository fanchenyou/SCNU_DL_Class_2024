import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(123)

# this example is modified from
# https://github.com/pytorch/examples/blob/main/time_sequence_prediction/train.py

T = 20  # period of sine wave
L = 1000  # sequence length
N = 100  # total training samples


def gen_sine_wave():
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    # period is 2pi*T = 40*3.14 = 124, check figure
    data = np.sin(x / 1.0 / T).astype('float64')
    return data


class WaveModel(nn.Module):
    def __init__(self):
        super(WaveModel, self).__init__()

        # initialize LSTM to have input_size=1 (sine value),
        # hidden_size=64, two-layer
        self.hidden_size = 64
        self.n_layers = 2
        self.lstm = nn.LSTM(input_size=1,
                            num_layers=self.n_layers,
                            hidden_size=self.hidden_size,
                            batch_first=True)
        # last layer is for regression
        self.linear = nn.Linear(self.hidden_size, 1)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))

    def forward(self, input, h0, c0):
        # Input is size [B_size, seq_len]
        # In training, seq_len=999, i.e., we feed LSTM x[0~998] to fit x[1~999]

        # Since LSTM needs 3-D input, make input [B, 999, 1]
        input = input.unsqueeze(-1)

        # run LSTM
        # output is [B, seq_len=999, hidden_size=64]
        output, (hn, cn) = self.lstm(input, (h0, c0))

        # make regression on next sine value with linear layer
        pred = self.linear(output)

        # remove last dim, since last dim=1, final pred size [B, 999]
        pred = pred.squeeze(-1)
        return pred

    def inference(self, input_prefix, h0, c0, predict_len=100):
        # In inference, given input, we predict the next predict_len values
        # Different from training, we need to
        # (1) feed input_prefix into LSTM, get last hidden h_t
        # (2) use last hidden h_t to predict next value y_t
        # (3) use h_t, y_t to make next prediction, h_t+1, y_t+1
        # repeat (2)-(3) for predict_len times

        ######################
        ###### Stage (1) #####
        ######################
        input_prefix = input_prefix.unsqueeze(-1)

        # run LSTM inference
        # output is [B, prefix_len=100, hidden_size=64]
        output, (hn, cn) = self.lstm(input_prefix, (h0, c0))

        # last hidden is extracted and need to keep dimension as [B, 1, hidden_size]
        # slicing trick -1: to keep dimension of 1
        prefix_last_hidden = output[:, -1:, :]
        # make regression on next sine value based on hn
        last_output = self.linear(prefix_last_hidden)

        ##########################
        ###### Stage (2)-(3) #####
        ##########################
        # here, we manually feed state into lstm step-by-step
        pred_step = []
        for p in range(predict_len):
            # TODO: use last hidden and output to feed lstm, update h_t+1, write a line of code here

            # TODO: update y_t+1 as last_output, write a line of code here

            # store y_t+1
            # currently last_output is dummy from encoder
            # you should update last_output as in previous TODO point
            pred_step.append(last_output)


        # make size [B, future_seq_len=100, 1]
        pred = torch.cat(pred_step, dim=1)
        # [B, 100]
        pred = pred.squeeze(-1)
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    #####################################
    ## generate or load sine wave data ##
    #####################################
    # load data and make training set
    data = gen_sine_wave()
    # [100, 1000]
    data = data.astype(np.float32)
    print('Generating sine wave data, check gen_sine_wave()')
    print('Data shape [B, L] is ', data.shape)
    assert data.shape[0] == N and data.shape[1] == L

    #########################
    # split train/test data
    #########################
    # input is (x0, x1, .., x_L-2)
    # target is one-shift of input, i.e., (x1, x2, ..., x_L-1)
    input = torch.from_numpy(data[:50, :-1])
    target = torch.from_numpy(data[:50, 1:])
    # val data is non-overlapping with training data
    val_input = torch.from_numpy(data[50:80, :-1])
    val_target = torch.from_numpy(data[50:80, 1:])
    # test data is non-overlapping with training data
    # test_input = torch.from_numpy(data[80:, :-1])
    # test_target = torch.from_numpy(data[80:, 1:])

    #########################
    # build the model
    #########################
    model = WaveModel()
    criterion = nn.MSELoss()

    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    # begin to train
    for i in range(opt.steps):
        print('STEP: ', i)

        ###################
        ### Training ######
        ###################
        # LBFGS needs to evaluate a process multiple times
        # pack the training process into a closure
        # https://pytorch.org/docs/stable/optim.html
        def closure():
            optimizer.zero_grad()
            h0, c0 = model.init_hidden(input.size(0))
            out = model(input, h0, c0)
            loss = criterion(out, target)
            # print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        #############################
        ### Validation/Inference ####
        #############################
        with torch.no_grad():
            # begin to complete a sine wave sequence
            pred_len = 100
            # for a sequence, the input is x0,...,x99
            val_input_prefix = val_input[:, :100]
            # we want to predict the next 100 values
            # so the ground-truth target is x100,...,x199
            val_output_target = val_input[:, 100:100 + pred_len]

            # init hidden
            h0, c0 = model.init_hidden(val_input_prefix.size(0))
            # call inference function, pred is [B, pred_future_len]
            pred = model.inference(val_input_prefix, h0, c0, predict_len=pred_len)
            # print(pred.size(), val_output_target.size())
            loss = criterion(pred, val_output_target)
            print('val loss:', loss.item())

        # draw the result
        if True:
            plt.figure(figsize=(30, 10))
            plt.title('Predict future values (green)\n(Dashlines are GT values)', fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            val_hist = val_input_prefix[0]
            val_pred = pred[0]
            val_gt = val_output_target[0]

            plt.plot(np.arange(pred_len), val_hist.cpu().numpy(), 'r', linewidth=2.0)
            plt.plot(np.arange(pred_len, pred_len + pred_len), val_pred.cpu().numpy(), 'g', linewidth=2.0)
            plt.plot(np.arange(pred_len, pred_len + pred_len), val_gt.cpu().numpy(), 'b' + ':', linewidth=2.0)

            if not os.path.isdir('out'):
                os.mkdir('out')
            plt.savefig('out/predict%d.png' % i)
            plt.close()
