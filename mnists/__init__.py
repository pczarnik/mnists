"""
MNISTs
======

Provides an easy access to MNIST-like datasets in a numpy format:
  1. MNIST
  2. FashionMNIST
  3. KMNIST
  4. EMNIST - divided into Balanced, ByClass, ByMerge, Digits and Letters
     subsets

Every dataset contains four numpy arrays:
  1. ``train_images`` of size ``(n_train_samples, width, height)``
  1. ``train_labels`` of size ``(n_train_samples,)``
  1. ``test_images`` of size ``(n_test_samples, width, height)``
  1. ``test_labels`` of size ``(n_test_samples,)``

All arrays are of type ``uint8``.

Example usage
-------------

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

"""

from ._emnist import EMNIST
from ._mnist import KMNIST, MNIST, FashionMNIST

FMNIST = FashionMNIST

__version__ = "0.3.1"
