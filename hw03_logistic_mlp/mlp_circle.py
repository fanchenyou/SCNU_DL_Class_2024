import torch
from sklearn.model_selection import train_test_split
from get_circle_quarter import train
from get_circle_quarter import get_circle, get_quarter


class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(MLPModel, self).__init__()
        # TODO: implement your 2-layer MLP here
        self.mlp_1 = None
        self.mlp_2 = None

    def forward(self, x):
        # TODO: Implement forward function here
        outputs = None
        return outputs


if __name__ == '__main__':

    ###############################
    ####     DO NOT MODIFY    #####
    ###############################
    input_dim = 2  # Two inputs x1 and x2
    output_dim = 1  # Two possible outputs

    # TODO: change this to 0/1 to try two cases
    use_circle_quarter = 0
    # TODO: you may modify hidden_size and learning_rate
    hidden_size = 2
    lr = 0.1

    ###############################
    ####     DO NOT MODIFY    #####
    ###############################
    if use_circle_quarter == 0:
        X, y = get_circle()
    elif use_circle_quarter == 1:
        X, y = get_quarter()
    else:
        assert 1 == 2, 'bad choice'
    # Get data, don't modify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # TODO: implement the MLPModel
    model = MLPModel(input_dim, hidden_size, output_dim)
    train(X_train, X_test, y_train, y_test, learning_rate=lr, model=model)

    for k, v in model.state_dict().items():
        print(k, v)
