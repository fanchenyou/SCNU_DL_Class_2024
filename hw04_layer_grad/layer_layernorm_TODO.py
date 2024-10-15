import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self, normal_shape, eps=1e-10):
        """Layer normalization layer

        Paper: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        Pytorch API Definition: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

        :param normal_shape: last dimensions except Batch, e.g., [B, x1, x2,...] -> normal shape [x1,x2,...]
        :param eps: Epsilon for calculating variance.
        """
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.normal_shape = normal_shape

    def forward(self, x):
        # TODO: implement layer normalization correctly
        y = x   # modify this
        return y



def main():
    ##################################
    ## DO NOT CHANGE ANY CODE BELOW ##
    ##################################

    # NLP Example, a batch of sentence embedding
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)

    # Pytorch official LN
    pt_layer_norm = nn.LayerNorm([sentence_length, embedding_dim])
    res1 = pt_layer_norm(embedding) # pytorch output

    # Your implementation
    my_layer_norm = LayerNormalization([sentence_length, embedding_dim], eps=1e-05,).float()
    res2 = my_layer_norm(embedding) # self-implementation output

    print(res1, '\n', res2)
    print('Check if they are same !')
    assert torch.allclose(res1,res2)



    # Image Example
    N, C, H, W = 2, 3, 5, 5
    input = torch.randn(N, C, H, W)

    # Pytorch official LN
    pt_layer_norm = nn.LayerNorm([C, H, W])
    res1 = pt_layer_norm(input) # pytorch output

    # Your implementation
    my_layer_norm = LayerNormalization([C, H, W], eps=1e-05,).float()
    res2 = my_layer_norm(input) # self-implementation output

    print(res1, '\n', res2)
    print('Check if they are same !')
    assert torch.allclose(res1,res2)
    




if __name__ == '__main__':
    main()