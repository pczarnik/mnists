import os
import tempfile
from typing import Optional
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlretrieve

import numpy as np

from .utils import check_file_integrity, read_idx_file

TEMPORARY_DIR = tempfile.gettempdir()


class MNIST:
    """
    MNIST Dataset
    http://yann.lecun.com/exdb/mnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from MNIST dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    classes : list[str]
        Class names.
    mirrors : list[str]
        List of urls where dataset is hosted.
    resources : dict[str, tuple[str, str]]
       Dictionary of data files with filename and md5 hash.
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

    default_dir = os.path.join(TEMPORARY_DIR, "mnist")

    def __init__(
        self,
        target_dir: Optional[str] = None,
        download: bool = True,
        force_download: bool = False,
        load: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        target_dir : str, default='/tmp/mnist/'
            Directory where all files exist or will be downloaded to (if `download` is True).
        download : bool, default=True
            If True and files don't exist in `target_dir`, downloads all files to `target_dir`.
        force_download : bool, default=False
            If True, downloads all files to `target_dir`, even if they exist there.
        load : bool, default=True
            If True, loads data from files in `target_dir`.
        """

        self.target_dir = self.default_dir if target_dir is None else target_dir

        self._train_images: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._test_images: Optional[np.ndarray] = None
        self._test_labels: Optional[np.ndarray] = None

        if download or force_download:
            self.download(force_download)

        if load:
            self.load()

    def download(self, force: bool = False) -> None:
        """
        Download files from mirrors and save to `target_dir`.

        Parameters
        ----------
        force : bool=False
            If True, download all files even if they exist.
        """

        os.makedirs(self.target_dir, exist_ok=True)

        for filename, md5 in self.resources.values():
            filepath = os.path.join(self.target_dir, filename)

            if not force and check_file_integrity(filepath, md5):
                continue

            self._download_file(filename, filepath)

    def load(self) -> None:
        """
        Load data from files in `target_dir`.
        """

        for key, (filename, md5) in self.resources.items():
            filepath = os.path.join(self.target_dir, filename)

            if not check_file_integrity(filepath, md5):
                raise RuntimeError(
                    f"Dataset '{key}' not found in '{filepath}' or MD5 "
                    "checksum is not valid. "
                    "Use download=True or .download() to download it"
                )

            data = read_idx_file(filepath)
            self._add_getter(key, data)

    def _add_getter(self, fn_name: str, data: np.ndarray) -> None:
        var_name = f"_{fn_name}"
        setattr(self, var_name, data)

        def getter() -> np.ndarray:
            return getattr(self, var_name)

        setattr(self, fn_name, getter)

    def _download_file(self, filename: str, filepath: str) -> None:
        for mirror in self.mirrors:
            url = urljoin(mirror, filename)
            try:
                print(f"Downloading {url}")
                urlretrieve(url, filepath)
                return
            except URLError as error:
                print(f"Failed to download {url} (trying next mirror):\n{error}")
                continue

        raise RuntimeError(f"Error downloading {filename}")


class FashionMNIST(MNIST):
    """
    Fashion-MNIST Dataset
    https://github.com/zalandoresearch/fashion-mnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from FashionMNIST dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    classes : list[str]
        Class names.
    mirrors : list[str]
        List of urls where dataset is hosted.
    resources : dict[str, tuple[str, str]]
       Dictionary of data files with filename and md5 hash.
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

    default_dir = os.path.join(TEMPORARY_DIR, "fmnist")


class KMNIST(MNIST):
    """
    Kuzushiji-MNIST Dataset
    https://github.com/rois-codh/kmnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from KMNIST dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    classes : list[str]
        Class names.
    mirrors : list[str]
        List of urls where dataset is hosted.
    resources : dict[str, tuple[str, str]]
       Dictionary of data files with filename and md5 hash.
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

    default_dir = os.path.join(TEMPORARY_DIR, "kmnist")
