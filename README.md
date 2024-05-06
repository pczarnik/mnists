# MNISTs: All MNIST-like datasets in one package

MNISTs provides an easy way to use MNIST and other MNIST-like datasets (e.g. FashionMNIST, KMNIST) in your numpy code.

MNISTs replicates the functionality of `torchvision.datasets.mnist` without the need to download dozens of dependencies.
MNISTs has only one dependency - `numpy`.


## Usage

Each dataset stores train/test images as numpy arrays of shape `(n_samples, img_height, img_width)` and train/test labels as numpy arrays of shape `(n_samples,)`.

MNIST example:
```python
>>> from mnists import MNIST
>>> mnist = MNIST()
>>> type(mnist.train_images())
<class 'numpy.ndarray'>
>>> mnist.train_images().dtype
dtype('uint8')
>>> mnist.train_images().min()
0
>>> mnist.train_images().max()
255
>>> mnist.train_images().shape
(60000, 28, 28)
>>> mnist.train_labels().shape
(60000,)
>>> mnist.test_images().shape
(10000, 28, 28)
>>> mnist.test_labels().shape
(10000,)
>>> mnist.classes[:3]
['0 - zero', '1 - one', '2 - two']
```

FashionMNIST example:
```python
from mnists import FashionMNIST
import matplotlib.pyplot as plt

fmnist = FashionMNIST()
plt.imshow(fmnist.train_images()[0], cmap='gray')
plt.title(fmnist.classes[fmnist.train_labels()[0]])
plt.axis('off')
plt.show()
```
![FashionMNIST example](https://raw.githubusercontent.com/pczarnik/mnists/main/imgs/fmnist_boot.png)

KMNIST example:
```python
from mnists import KMNIST
import matplotlib.pyplot as plt

kmnist = KMNIST()
plt.imshow(
    kmnist.test_images()[:256]
        .reshape(16, 16, 28, 28)
        .swapaxes(1, 2)
        .reshape(16 * 28, -1),
    cmap='gray')
plt.axis('off')
plt.show()
```
![KMNIST example](https://raw.githubusercontent.com/pczarnik/mnists/main/imgs/kmnist_256.png)


## Installation

Install `mnists` from [PyPi](https://pypi.org/project/mnists):
```
pip install mnists
```
or from source:
```
pip install -U git+https://github.com/pczarnik/mnists
```

The only requirements for MNISTs are `numpy>=1.22` and `python>=3.9`.


## Acknowledgments

The main inspirations for MNISTs were [`mnist`](https://github.com/datapythonista/mnist) and [`torchvision.datasets.mnist`](https://github.com/pytorch/vision).
