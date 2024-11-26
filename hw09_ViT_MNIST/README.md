# ViT: Visual Transformer Networks

* This repository is modified from https://github.com/peluche/ViT
* This provides you a toy example of implementing a tiny ViT model and training on MNIST data
* Recommend to run on a 4060+ GPU in Colab or AutoDL (highly recommend)
* Or, you can borrow a checkpoint file from your friend and skip the training step

## Step 1: download data from
https://github.com/fgnt/mnist/tree/master

Uncompress gz and put in folder like
```
data/MNIST/raw/
 t10k-images-idx3-ubyte
 t10k-labels-idx1-ubyte
 train-images-idx3-ubyte 
 train-labels-idx1-ubyte
```
or turn on download=True in ipynb file (slow for downloading via torch dataset API)

## Step 2: go through vit.ipynb step-by-step
Answer all TODO explanations