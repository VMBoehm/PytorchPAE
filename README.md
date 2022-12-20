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
- [optuna] (https://optuna.readthedocs.io/en/stable/)
- torchsummary

Installation:
```sh
git clone https://github.com/VMBoehm/PytorchPAE.git
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

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2006.05479):

```
@ARTICLE{PAE,
       author = {{B{\"o}hm}, Vanessa and {Seljak}, Uro{\v{s}}},
       title={Probabilistic Autoencoder},
       journal={Transactions on Machine Learning Research},
       year={2022},
       url={https://openreview.net/forum?id=AEoYjvjKVA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200605479B},
       doi = {10.48550/ARXIV.2006.05479}
}
```
