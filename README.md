# Deep K Mean Clustering

This repository contains implementation of [Deep k-Means: Jointly clustering with k-Means and learning representations](https://arxiv.org/abs/1806.10069)

## How to run

```bash
python3 train_deep_k_means_mnist.py
```
The script launches comparison of Scikit-learn's implementation of K-Means with Deep K-Means.
Test task is to separate MNIST data on 10 clusters. 
Used metrics are clustering accuracy and normalized mutual info score.

_*Expected output:*_

```
K-means
   ACC:  0.5134333333333333
   NMI:  0.4905763104545929
Deep K-means
   ACC:  0.84575
   NMI:  0.7918360450505724
```

Where ACC stands for accuracy and NMI is for normalized mutual info.

## Enviroment specifications
* tensorflow==2.3.1
* numpy==1.18.1
* scikit-learn==0.21.3
* scipy==1.4.1
