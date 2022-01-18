# PytorchPAE
A highly modular pytorch package for easy and fast implementation and training of a [probabilistic autoencoder](https://github.com/VMBoehm/PAE).

The current version features 
- support for fully connected and convolutional AE architectures for 1D and 2D data
- a Sliced Iterative Normalizing Flow as density estimator
- an example for how to automatically optimize the network architecture with Optuna 
- a maximally modular design that allows the user to add custom datasets, architectures and loss functions



## Installation and Requirements

Requirements: 
- pytorch 1.8.0
- [sinf](https://github.com/biweidai/SINF)

Optional:
- optuna 
- torchsummary

Installation:
```sh
git clone https://github.com/VMBoehm/PytorchPAE
```
```sh
cd PytorchPAE
```
```sh
pip install -e . 
```
(follow the same steps to install sinf)


## Getting started

A tutorial for how to use his package is provided [here](https://github.com/VMBoehm/PytorchPAE/blob/main/notebooks/Tutorial.ipynb)


