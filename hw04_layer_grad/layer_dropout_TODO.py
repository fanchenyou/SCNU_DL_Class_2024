import torch
import torch.nn as nn

# Refer to Caffe Dropout implementation
# https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cpp
class MyDropout(nn.Module):
    def __init__(self, p: float = 0.05):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        # p is probability of zeros -- to be dropped
        # thus (1-p) is prob. of unchanged elements
        self.p = p
        self.scale = 1.0/(1-self.p)

    def forward(self, X):
        #######################################
        ## DO NOT CHANGE ANY CODE in forward ##
        #######################################
        # TODO: Read forward function, explain what dropout does
        if self.training:
            # in training, randomly sample prob. (e.g., 5%) elements to be zero (dropped)
            # then the rest 1-prob. elements of X to remain same
            # First, we construct a mask, e.g., if p=0.05, then 95% of the mask values are 1
            # the 5% values are 0, which drop the elements
            # check https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cpp#L39
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            # use the Binomial distribution to sample a binary mask
            # indicating which elements to drop (mask[i,j]=0) and retain (mask[i,j]=1)
            self.mask = binomial.sample(X.size())
            # dropout X
            X_masked = X * self.mask
            # then we have to scale the element values to be 1/(1-prob), to make X_masked roughly sum to original X
            # e.g., if p=0.5, you randomly dropout half elements in X, the left half of X should be made values as 2X
            X_scale = X_masked * self.scale
            return  X_scale
        
        # in inference (validation/testing), no need to scale
        return X

    def backward_manual(self, delta_X_top):
        # TODO: implement backward function
        if self.training:
            delta_X_bottom = delta_X_top  # TODO, modify this
        else:
            delta_X_bottom = delta_X_top  # TODO, should modify this or not ?
        return delta_X_bottom



def main():
    ##################################
    ## DO NOT CHANGE ANY CODE BELOW ##
    ##     Explain TODO  places     ##
    ##################################

    '''
    Let y = dropout(x) be prediction.
    Let the true value is 1.
    Then the loss L = (y-1.0)^2
    Delta_X = dL/dx = dL/dy * dy/dx = 2(y-1.0) * dy/dx
    Note that dy/dx is the backward_manual implemented by you
    We now compare your dy/dx with torch.autograd
    '''

    # fix seed, do not modify
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # declare you dropout
    dropout = MyDropout(p=0.5)
    # turn on training
    dropout.training = True

    # test case, print out the input x
    x = torch.arange(0,9, dtype=torch.float32, requires_grad=True).view(3,3)


    # forward
    print('Input ', x)
    y = dropout.forward(x)
    print(' - dropout forward:\n', y)

    # let's assume a toy model, with y = dropout(x), loss = 0.5* y^2
    loss_y_0 = 0.5*(y**2)
    # sum the loss to a scala
    loss_y = torch.sum(loss_y_0)

    # TODO: explain the result, what is dloss/dy
    y_diff = torch.autograd.grad(loss_y, y, retain_graph=True)[0]
    print('Loss y gradient is \n', y_diff)

    # TODO: explain why dropout manual backward function you implemented is to compute dy/dx (here use variable dx)
    dx = dropout.backward_manual(y_diff)
    print('Dropout manual backward:\n', dx)

    # TODO: explain the result, use torch autograd to get x's gradient
    dx2 = torch.autograd.grad(loss_y, x, retain_graph=True)[0]
    print('Dropout auto backward:\n', dx2)

    # TODO: explain why dx=dx2, use chain rule to explain
    # hint: y = Dropout(x), loss=0.5*y^2, by chain-rule, dy/dx = ?


    # the assertions should be correct after your implementation
    assert torch.allclose(dx, dx2), 'the assertions should be correct after your implementation'



if __name__ == '__main__':
    main()