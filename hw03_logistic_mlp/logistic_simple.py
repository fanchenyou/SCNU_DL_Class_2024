import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from get_circle_quarter import train


class LogisticRegressionSimple(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionSimple, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def get_BiNormal():
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    mu1 = -1.5 * torch.ones(2)
    mu2 = 1.5 * torch.ones(2)

    sigma1 = torch.eye(2) * 0.6
    sigma2 = torch.eye(2) * 1.2

    m1 = torch.distributions.MultivariateNormal(mu1, sigma1)
    m2 = torch.distributions.MultivariateNormal(mu2, sigma2)

    x1 = m1.sample((1000,))
    x2 = m2.sample((1000,))

    y1 = np.zeros(x1.size(0))
    y2 = np.ones(x2.size(0))

    X = torch.cat([x1, x2], dim=0)
    Y = np.concatenate([y1, y2])

    return X, Y



if __name__ == '__main__':
    X, y = get_BiNormal()
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    input_dim = 2  # Two inputs x1 and x2
    output_dim = 1  # Two possible outputs
    model = LogisticRegressionSimple(input_dim, output_dim)
    train(X_train, X_test, y_train, y_test, learning_rate=0.01, model=model)
