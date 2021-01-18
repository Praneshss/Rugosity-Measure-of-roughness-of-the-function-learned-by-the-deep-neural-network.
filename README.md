# Rugosity: Measure of roughness of the function learned by the deep neural network.
Implicit Rugosity Regularization via Data Augmentation


## Introduction
This repository contains the Keras implementation of the rugosity measure, proposed in the work  [Implicit Rugosity Regularization via Data Augmentation](https://arxiv.org/pdf/1905.11639.pdf). The measure is demonstrated for MNIST dataset in this code.


## Folder Structure
The folder structure of this repo is mentioned below:
```
Rugosity
├── analysis/
├── mnist_rugosity.py
```

##  Analysis
We analyse the impact on rugosity measure against
1. validation accuracy
2.  different initializations

This analysis code is located in folder ```analysis```



##  Using rugosity for regularization
Please refer to repo [https://github.com/RandallBalestriero/Hessian](https://github.com/RandallBalestriero/Hessian) for rugosity based regularization.




## Acknowledgement
Thanks to Randall and Daniel for assisting during implementation.




