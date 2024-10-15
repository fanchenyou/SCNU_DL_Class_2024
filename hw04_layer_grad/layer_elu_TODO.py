import torch
import torch.nn as nn


class MyELU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(MyELU, self).__init__()
        if alpha < 0:
            raise ValueError("negative_slope should be >0, " "but got {}".format(alpha))
        self.alpha = alpha

    def forward(self, X_bottom):
        #######################################
        ## DO NOT CHANGE ANY CODE in forward ##
        #######################################
        # record the mask
        # TODO: Explain in hw why this is important ?
        self.X_exp = torch.exp(X_bottom)
        X_neg = self.alpha * (self.X_exp-1)
        self.mask = (X_bottom > 0)
        # slope is 1 for positive values
        X_top = torch.zeros_like(X_bottom)
        # slope is alpha for negative values
        X_top[self.mask] = X_bottom[self.mask]
        X_top[~self.mask] = X_neg[~self.mask]
        return X_top

    def backward_manual(self, delta_X_top):
        # TODO: implement backward function
        # hint: you may re-use the recorded self.mask and self.X_exp in forward() function
        # take the gradient and compute the gradient formula
        # TODO:  re-use the recorded mask in forward() function, why this is important? Explain.
        delta_X_bottom = delta_X_top # replace this dummy line to your code
        return delta_X_bottom


def main():
    ##################################
    ## DO NOT CHANGE ANY CODE BELOW ##
    ##      Explain TODO  places    ##
    ##################################
    '''
    Let y = elu(x) be prediction.
    Let the true value is 1.
    Then the loss L = (y-1.0)^2
    Delta_X_bottom = dL/dx = dL/dy * dy/dx = 2(y-1.0) * dy/dx
    Note that dL/dy is actually the delta_X_top;
    Note that dy/dx is the gradient of ELU layer, i.e.,
     the backward_manual implemented by you
    We can verify this by comparing your dy/dx with torch.autograd
    '''

    # test case as input
    x = torch.arange(-4, 5, dtype=torch.float32, requires_grad=True).view(3, 3)

    # ========================
    # === MyELU forward ======
    # ========================
    my_elu = MyELU(alpha=1.0)

    # forward
    print('Input ', x)
    y = my_elu.forward(x)
    print(' - my_elu forward:\n', y, y.size())

    # let's assume a toy model, with y = elu(x), loss = 0.5* y^2
    loss_y_0 = 0.5 * (y ** 2)
    # sum the loss to a scala
    loss_y = torch.sum(loss_y_0)

    # TODO: explain the result, what is dloss/dy
    y_diff = torch.autograd.grad(loss_y, y, retain_graph=True)[0]
    print('Loss y gradient is \n', y_diff)

    # Now we use two ways to compute dloss_y / dx, they should be the same

    # =============================
    # ==== My ELU backward ========
    # =============================
    if True:
        # TODO: explain the result, calculate the gradient with manual backward function you implemented
        dx = my_elu.backward_manual(y_diff)
        print('MyELU manual backward:\n', dx)

        # TODO: explain the result, use torch autograd to get x's gradient
        dx2 = torch.autograd.grad(loss_y, x, retain_graph=True)[0]
        print('MyELU auto backward:\n', dx2)

        # TODO: explain why dx=dx2, use chain rule to compute, then compare
        # hint: y = Elu(x), loss=0.5*y^2, by chain-rule, dy/dx = ?

        print('They should be same !')
        assert torch.allclose(dx, dx2)

    # =========================
    # ===== Torch ELU  ========
    # =========================
    if True:
        print('\n========= Below is Pytorch Implementation ===========')
        # TODO: here we directly use Pytorch ELU. Explain, Should be y==y3? dx==dx3? Explain
        torch_elu = torch.nn.ELU(alpha=1.0)
        y3 = torch_elu(x)
        print('Torch ELU forward:\n', y3)
        loss_y3 = torch.sum(0.5 * (y3 ** 2))
        dx3 = torch.autograd.grad(loss_y3, x, retain_graph=True)[0]
        print('Torch ELU manual backward:\n', dx3)

        # the assertions should be correct after your implementation
        assert torch.allclose(y, y3)
        assert torch.allclose(dx, dx3), 'the assertions should be correct after your implementation'


if __name__ == '__main__':
    main()
