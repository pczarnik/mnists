from .dataset import IdxDataset


class MNIST(IdxDataset):
    """
    MNIST Dataset
    http://yann.lecun.com/exdb/mnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    classes : list[str]
        Class names.
    mirrors : list[str]
        List of urls where dataset is hosted.
    resources : dict[str, tuple[str, str]]
       Dictionary of data files with filename and md5 hash.

    Usage
    -----
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

    Citation
    --------
    @article{lecun-98,
      author={Lecun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
      journal={Proceedings of the IEEE},
      title={Gradient-based learning applied to document recognition},
      year={1998},
      volume={86},
      number={11},
      pages={2278-2324},
      doi={10.1109/5.726791}
    }
    """

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]

    resources = {
        "train_images": (
            "train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        "train_labels": (
            "train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        "test_images": (
            "t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        "test_labels": (
            "t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    }


class FashionMNIST(IdxDataset):
    """
    Fashion-MNIST Dataset
    https://github.com/zalandoresearch/fashion-mnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    classes : list[str]
        Class names.
    mirrors : list[str]
        List of urls where dataset is hosted.
    resources : dict[str, tuple[str, str]]
       Dictionary of data files with filename and md5 hash.

    Usage
    -----
    >>> from mnists import FashionMNIST
    >>> fmnist = FashionMNIST()
    >>> fmnist.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @online{xiao2017/online,
      author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
      title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking
                      Machine Learning Algorithms},
      date         = {2017-08-28},
      year         = {2017},
      eprintclass  = {cs.LG},
      eprinttype   = {arXiv},
      eprint       = {cs.LG/1708.07747},
    }
    """

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    mirrors = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    ]

    resources = {
        "train_images": (
            "train-images-idx3-ubyte.gz",
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        ),
        "train_labels": (
            "train-labels-idx1-ubyte.gz",
            "25c81989df183df01b3e8a0aad5dffbe",
        ),
        "test_images": (
            "t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554ea6380940ec79",
        ),
        "test_labels": (
            "t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a12a480ee83cd310",
        ),
    }


class KMNIST(IdxDataset):
    """
    Kuzushiji-MNIST Dataset
    https://github.com/rois-codh/kmnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    classes : list[str]
        Class names.
    mirrors : list[str]
        List of urls where dataset is hosted.
    resources : dict[str, tuple[str, str]]
       Dictionary of data files with filename and md5 hash.

    Usage
    -----
    >>> from mnists import KMNIST
    >>> kmnist = KMNIST()
    >>> kmnist.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @online{clanuwat2018deep,
      author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto
                      and Alex Lamb and Kazuaki Yamamoto and David Ha},
      title        = {Deep Learning for Classical Japanese Literature},
      date         = {2018-12-03},
      year         = {2018},
      eprintclass  = {cs.CV},
      eprinttype   = {arXiv},
      eprint       = {cs.CV/1812.01718},
    }
    """

    classes = [
        "o",
        "ki",
        "su",
        "tsu",
        "na",
        "ha",
        "ma",
        "ya",
        "re",
        "wo",
    ]

    mirrors = [
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    ]

    resources = {
        "train_images": (
            "train-images-idx3-ubyte.gz",
            "bdb82020997e1d708af4cf47b453dcf7",
        ),
        "train_labels": (
            "train-labels-idx1-ubyte.gz",
            "e144d726b3acfaa3e44228e80efcd344",
        ),
        "test_images": (
            "t10k-images-idx3-ubyte.gz",
            "5c965bf0a639b31b8f53240b1b52f4d7",
        ),
        "test_labels": (
            "t10k-labels-idx1-ubyte.gz",
            "7320c461ea6c1c855c0b718fb2a4b134",
        ),
    }
