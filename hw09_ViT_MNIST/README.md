# ViT: Visual Transformer Networks

Toy implementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf) on MNIST.

## Chunking Images
```python
tiles = einops.rearrange(images, 'b c (h t1) (w t2) -> b (h w) c t1 t2', t1=tile_size, t2=tile_size)
```
![tiles](imgs/tiles.png)

## Attention
![attention_subject](imgs/attention_subject.png)
![attention_0](imgs/attention_0.png)
![attention_1](imgs/attention_1.png)

## Positional Embeddings
![learned_positional_embeddings](imgs/learned_positional_embeddings.png)

## Misslabelled Images
![misslabelled](imgs/misslabelled.png)

## Confusion Matrix
![confusion_matrix](imgs/confusion_matrix.png)

## Heatmap
![heatmap](imgs/heatmap.png)
