import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_circle():
    np.random.seed(123)
    X = np.random.normal(0, 1, size=(1000, 2))
    y = np.array((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0, dtype='int')
    return X, y


def get_quarter():
    np.random.seed(123)
    X = np.random.normal(0, 1, size=(1000, 2))

    y = np.logical_and(np.logical_and((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0, X[:, 0] > 0), X[:, 1] > 0)

    # y = np.array((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0, dtype='int')
    y = np.array(y, dtype='int')
    return X, y


def vis_circle_quarter(X, y, X2, y2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Vertically stacked subplots')

    ax1.scatter(X[y <= 0.5, 0], X[y <= 0.5, 1], c='b')
    ax1.scatter(X[y > 0.5, 0], X[y > 0.5, 1], c='g')
    ax1.set_title('Circle')

    ax2.scatter(X2[y2 <= 0.5, 0], X2[y2 <= 0.5, 1], c='b')
    ax2.scatter(X2[y2 > 0.5, 0], X2[y2 > 0.5, 1], c='g')
    ax2.set_title('Quarter')

    plt.show()



def train(X_train, X_test, y_train, y_test, learning_rate=0.01, model=None):
    epochs = 20000

    # Binary Cross-Entropy
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)


    iter = 0

    for epoch in tqdm(range(int(epochs)+1), desc='Training Epochs'):
        labels = y_train
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(torch.squeeze(outputs), labels)  # BCE Loss
        loss.backward()  # Computes the gradient of the given tensor w.r.t. graph leaves

        optimizer.step()  # Updates weights and biases with the optimizer (SGD)

        iter += 1
        if iter % 500 == 0:
            # calculate Accuracy
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)

                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_test.size(0)
                correct_test += np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test / total_test

                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct / total

                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    vis_prediction(X_test, y_test, predicted_test)


def vis_prediction(X_test, y_test, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


    ax1.scatter(X_test[y_test <= 0.5, 0], X_test[y_test <= 0.5, 1], c='b')
    ax1.scatter(X_test[y_test > 0.5, 0], X_test[y_test > 0.5, 1], c='g')
    ax1.set_title('True data')

    ax2.scatter(X_test[y_pred <= 0.5, 0], X_test[y_pred <= 0.5, 1], c='b')
    ax2.scatter(X_test[y_pred > 0.5, 0], X_test[y_pred > 0.5, 1], c='g')

    error_1 = np.logical_and(y_pred <= 0.5, y_test > 0.5)
    error_2 = np.logical_and(y_pred > 0.5, y_test <= 0.5)
    ax2.scatter(X_test[error_1, 0], X_test[error_1, 1], c='r')
    ax2.scatter(X_test[error_2, 0], X_test[error_2, 1], c='orange')

    ax2.set_title('Pred data and errors')

    plt.show()