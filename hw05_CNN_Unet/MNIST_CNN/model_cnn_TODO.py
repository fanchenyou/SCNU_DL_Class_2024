import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2)

        # TODO: uncomment this, choose proper layers, modify their parameters slightly if necessarily
        if 1 == 2:
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.3)
            self.dropout2 = nn.Dropout(0.3)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        # TODO: remove this dummy fc layer, use fc1 and/or fc2 as output
        self.fc_dummy = nn.Linear(5408, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # TODO: remove the following dummy lines
        # change to your layers forward function
        # make sure the output is [Batch_size, 10] as there are 10 classes to predict
        x = torch.flatten(x, 1)
        out = self.fc_dummy(x)
        # print(out.size())
        assert out.ndim == 2 and out.size(0) == x.size(0) and out.size(1) == 10

        return out
