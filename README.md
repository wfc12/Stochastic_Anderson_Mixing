# Stochastic_Anderson_Mixing
Implementation of the stochastic Anderson mixing method for nonconvex stochastic optimization.

## Usage

PyTorch1.4.0 (except that using PyTorch1.8.0 for LSTM on Penn TreeBank)

1. Put optim/adasam.py, optim/adasamvr.py, and optim/padasam.py to the optim directory of PyTorch.

2. Modify __init__.py by adding

```
from .adasam import AdaSAM
del adasam
from .padasam import pAdaSAM
del padasam
from .adasamvr import AdaSAMVR
del adasamvr
```

3. See the readme.txt in each sub-directory for running an example.

## Experiments

Experiments in this repository were based on the following repositories.

MNIST: [mnist](https://github.com/pytorch/examples/blob/master/mnist)

ResNet: [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)

LSTM: [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)

## Additional

If you find this repository useful for your research, please consider citing our work:

```
@inproceedings{WeiBL21SAM,
  author    = {Fuchao Wei and Chenglong Bao and Yang Liu}, 
  title     = {Stochastic Anderson Mixing for Nonconvex Stochastic Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  pages     = {22995--23008},
  volume = {34},
  year      = {2021}
}
```
