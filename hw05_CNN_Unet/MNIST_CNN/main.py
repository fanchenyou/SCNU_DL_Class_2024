import os
import argparse
import numpy as np
from torch.utils import data
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from model_cnn_TODO import Net
from PIL import Image


def train(args, model, device, train_loader, optimizer, epoch):
    # TODO: explain the function of train()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Lr:{:.4f} Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), optimizer.param_groups[0]['lr'], loss.item()))


# phase indicate validating or testing
# note: during training, you can only validate
def test(model, device, test_loader, phase='validate'):
    # TODO: explain the function of eval()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            # TODO: explain the function of argmax
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n {}: Average loss: {:.4f}, Accuracy: {}/{} \t ({:.0f}%)\n'.format(
        phase,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# read source https://pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html#MNIST
class mnist_dataset(data.Dataset):
    def __init__(self, data: np.array, label: np.array, transform: transforms):
        super(mnist_dataset, self).__init__()
        self.data = data
        self.label = label.astype(np.int64)
        self.transform = transform

        # sanity check
        assert self.data.shape[0] == self.label.shape[0]

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __getitem__(self, index: int):
        image = self.data[index, :]
        label = self.label[index, 0]

        # TODO: explain each line of reading in a gray image
        # use Image to convert to image and apply normalization
        image = Image.fromarray(image, mode="L")
        image = self.transform(image)
        # image [1,784] convert to a 3-D matrix of size (1,28,28)
        # the first channel 1 says its gray image with channel 1
        # if you process RGB image, that will be 3-channel instead 1
        image = image.squeeze().view(1, 28, 28)
        return image, label


def load_mnist_data(args):
    # download mnist numpy data files from
    # https://www.kaggle.com/datasets/sivasankaru/mnist-npy-file-dataset?resource=download
    train_labels = np.load('data/train_labels.npy')
    train_images = np.load('data/train_images.npy')
    test_labels = np.load('data/test_labels.npy')
    test_images = np.load('data/test_images.npy')

    # (10000, 1) (60000, 1)
    print(test_labels.shape, train_labels.shape)
    # (10000, 784) (60000, 784)
    print(test_images.shape, train_images.shape)

    # TODO: explain transforms of images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = mnist_dataset(train_images, train_labels, transform)
    test_data = mnist_dataset(test_images, test_labels, transform)

    train_size = int(0.8 * len(train_images))
    validate_size = len(train_images) - train_size
    print('Split training data into %d for training and %d for validation' % (train_size, validate_size))

    train_dataset, validate_dataset = torch.utils.data.random_split(train_data, [train_size, validate_size])
    print(len(train_dataset), len(validate_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print(len(train_loader), len(validate_loader), len(test_loader))
    return train_loader, validate_loader, test_loader


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--type', type=str, default='SGD', choices=['SGD', 'ADAM'],
                        help='use SGD or ADAM')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # if you have a GPU, turn this on
    use_cuda = args.cuda and torch.cuda.is_available()

    # if you have a MAC M1/2 chip, turn this on
    use_mps = False  # args.mps or torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print('The device is', device)

    seed = 47
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load mnist data
    train_loader, validate_loader, test_loader = load_mnist_data(args)

    model = Net().to(device)

    if args.type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise "This model type is not implemented !!"

    scheduler = StepLR(optimizer, step_size=8, gamma=args.gamma)

    # training phase, use validation set to check performance
    # TODO: record the validation accuracy in each epoch
    # use Matplot to draw the accuracy changes over each epoch
    # compare SGD, ADAM convergence and explain which and why is better
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, validate_loader, phase='Validate')
        scheduler.step()

    # testing phase, use test set to evaluate the final performance
    print('========= Final Testing =============')
    test(model, device, test_loader, phase='Test')

    # save model
    if args.save_model:
        if not os.path.isdir('output'):
            os.mkdir('output')
        torch.save(model.state_dict(), "output/mnist_%s.pt" % (args.type,))


if __name__ == '__main__':
    main()
