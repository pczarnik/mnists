# MNISTs: All MNIST-like datasets in one package

MNISTs provides an easy way to use MNIST and other MNIST-like datasets (e.g. FashionMNIST, KMNIST) in your numpy code.

MNISTs replicates what torchvision.datasets.mnist provides, but without all the dependencies - the only dependency is numpy.


## Usage

Each dataset contains train/test images and labels in a numpy array of shape `(n_samples, img_height, img_width)`.

MNIST example:
```python
>>> from mnists import MNIST
>>> mnist = MNIST()
>>> type(mnist.train_images)
<class 'numpy.ndarray'>
>>> mnist.train_images.dtype
dtype('uint8')
>>> mnist.train_images.min()
0
>>> mnist.train_images.max()
255
>>> mnist.train_images.shape
(60000, 28, 28)
>>> mnist.train_labels.shape
(60000,)
>>> mnist.test_images.shape
(10000, 28, 28)
>>> mnist.test_labels.shape
(10000,)
>>> mnist.classes
['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
```

FashionMNIST example:
```python
from mnists import FashionMNIST
import matplotlib.pyplot as plt

fmnist = FashionMNIST()
plt.imshow(fmnist.train_images[0], cmap='gray')
plt.title(fmnist.classes[fmnist.train_labels[0]])
plt.axis('off')
plt.show()
```
![FashionMNIST example](https://github.com/pczarnik/mnists/imgs/fmnist_boot.png)

KMNIST example:
```python
from mnists import KMNIST
import matplotlib.pyplot as plt

kmnist = KMNIST()
plt.imshow(
    kmnist.test_images[:256]
        .reshape(16, 16, 28, 28)
        .swapaxes(1, 2)
        .reshape(16 * 28, -1),
    cmap='gray')
plt.axis('off')
plt.show()
```
![KMNIST example](https://github.com/pczarnik/mnists/imgs/kmnist_256.png)


## Installation

Install from PyPi:
```
pip install mnists
```
or from source:
```
pip install -U git+https://github.com/pczarnik/mnists
```

The only requirements for MNISTs are `numpy>=1.22` and `python>=3.9`.
