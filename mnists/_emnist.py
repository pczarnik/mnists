import os
from typing import Optional

import numpy as np

from ._mnist import MNIST, TEMPORARY_DIR
from .utils import check_file_integrity, extract_from_zip


class EMNIST(MNIST):
    """
    EMNIST Dataset
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    Balanced, ByClass, ByMerge, Digits, Letters : class
        Classes containing splits of EMNIST dataset.
    """

    mirrors = [
        "https://biometrics.nist.gov/cs_links/EMNIST/",
    ]

    resources = {"gzip": ("gzip.zip", "58c8d27c78d21e728a6bc7b3cc06412e")}

    def __init__(
        self,
        target_dir: Optional[str] = None,
        download: bool = True,
        force_download: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        target_dir : str, default='/tmp/emnist/'
            Directory where zip exists or will be downloaded to (if `download` is True).
        download : bool, default=True
            If True and zip doesn't exist in `target_dir`, downloads zip to `target_dir`.
        force_download : bool, default=False
            If True, downloads zip to `target_dir`, even if it exists there.
        """
        self.target_dir = (
            os.path.join(TEMPORARY_DIR, type(self).__name__)
            if target_dir is None
            else target_dir
        )

        self.Balanced = self._create_split(Balanced)
        self.ByClass = self._create_split(ByClass)
        self.ByMerge = self._create_split(ByMerge)
        self.Digits = self._create_split(Digits)
        self.Letters = self._create_split(Letters)

        if download or force_download:
            self.download(force_download)

    def _create_split(self, split_cls: type["_Split"]) -> type["_Split"]:
        split_cls.default_base_dir = self.target_dir
        split_cls.default_zip_filepath = os.path.join(self.target_dir, "gzip.zip")
        split_cls.zip_md5 = self.resources["gzip"][1]
        return split_cls


class _Split(MNIST):
    default_base_dir = os.path.join(TEMPORARY_DIR, "emnist")
    default_zip_filepath = os.path.join(default_base_dir, "gzip.zip")
    zip_md5 = None

    def __init__(
        self,
        target_dir: Optional[str] = None,
        zip_filepath: Optional[str] = None,
        unzip: bool = True,
        force_unzip: bool = False,
        load: bool = True,
        transpose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        target_dir : str, default='/tmp/emnist/<split_name>/'
            Directory where all files exist or will be unzipped to (if `unzip` is True).
        zip_filepath : str, default='/tmp/emnist/gzip.zip'
            Filepath to zip file containing all EMNIST split files.
        unzip : bool, default=True
            If True and files don't exist in `target_dir`, unzips all files to `target_dir`.
        force_unzip : bool, default=False
            If True, unzips all files to `target_dir`, even if they exist there.
        load : bool, default=True
            If True, loads data from files in `target_dir`.
        transpose : bool, default=True
            If True, transposes train and test images.
        """

        self.target_dir = (
            os.path.join(self.default_base_dir, type(self).__name__)
            if target_dir is None
            else target_dir
        )

        self.zip_filepath = (
            self.default_zip_filepath if zip_filepath is None else zip_filepath
        )

        self._train_images: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._test_images: Optional[np.ndarray] = None
        self._test_labels: Optional[np.ndarray] = None

        if unzip or force_unzip:
            self.unzip_files(force_unzip)

        if load:
            self.load(transpose)

    def unzip_files(self, force: bool = False) -> None:
        """
        Unzip files from `zip_filepath` to `target_dir`.

        Parameters
        ----------
        force : bool=False
            If True, unzips all files even if they exist.
        """

        os.makedirs(self.target_dir, exist_ok=True)
        if not check_file_integrity(self.zip_filepath, self.zip_md5):
            raise RuntimeError(
                f"Zip file '{self.zip_filepath}' doesn't exists or its MD5"
                "checksum is not valid. "
                "Use EMNIST(download=True) or emnist.download() to download it"
            )

        for filename, md5 in self.resources.values():
            filepath = os.path.join(self.target_dir, filename)

            if not force and check_file_integrity(filepath, md5):
                continue

            extract_from_zip(self.zip_filepath, filename, self.target_dir)

    def load(self, transpose: bool = True) -> None:
        """
        Load data from files in `target_dir` and transpose images (by default).

        Parameters
        ----------
        transpose : bool=True
            If True, transposes train and test images.
        """

        super().load()
        if transpose:
            self._transpose_images()

    def _transpose_images(self) -> None:
        self._train_images = np.moveaxis(self._train_images, -2, -1)
        self._test_images = np.moveaxis(self._test_images, -2, -1)


class Balanced(_Split):
    """
    EMNIST Balanced
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from EMNIST Balanced dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    """

    resources = {
        "train_images": (
            "emnist-balanced-train-images-idx3-ubyte.gz",
            "4041b0d6f15785d3fa35263901b5496b",
        ),
        "train_labels": (
            "emnist-balanced-train-labels-idx1-ubyte.gz",
            "7a35cc7b2b7ee7671eddf028570fbd20",
        ),
        "test_images": (
            "emnist-balanced-test-images-idx3-ubyte.gz",
            "6818d20fe2ce1880476f747bbc80b22b",
        ),
        "test_labels": (
            "emnist-balanced-test-labels-idx1-ubyte.gz",
            "acd3694070dcbf620e36670519d4b32f",
        ),
    }


class ByClass(_Split):
    """
    EMNIST ByClass
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from EMNIST ByClass dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    """

    resources = {
        "train_images": (
            "emnist-byclass-train-images-idx3-ubyte.gz",
            "712dda0bd6f00690f32236ae4325c377",
        ),
        "train_labels": (
            "emnist-byclass-train-labels-idx1-ubyte.gz",
            "ee299a3ee5faf5c31e9406763eae7e43",
        ),
        "test_images": (
            "emnist-byclass-test-images-idx3-ubyte.gz",
            "1435209e34070a9002867a9ab50160d7",
        ),
        "test_labels": (
            "emnist-byclass-test-labels-idx1-ubyte.gz",
            "7a0f934bd176c798ecba96b36fda6657",
        ),
    }


class ByMerge(_Split):
    """
    EMNIST ByMerge
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from EMNIST ByMerge dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    """

    resources = {
        "train_images": (
            "emnist-bymerge-train-images-idx3-ubyte.gz",
            "4a792d4df261d7e1ba27979573bf53f3",
        ),
        "train_labels": (
            "emnist-bymerge-train-labels-idx1-ubyte.gz",
            "491be69ef99e1ab1f5b7f9ccc908bb26",
        ),
        "test_images": (
            "emnist-bymerge-test-images-idx3-ubyte.gz",
            "8eb5d34c91f1759a55831c37ec2a283f",
        ),
        "test_labels": (
            "emnist-bymerge-test-labels-idx1-ubyte.gz",
            "c13f4cd5211cdba1b8fa992dae2be992",
        ),
    }


class Digits(_Split):
    """
    EMNIST Digits
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from EMNIST Digits dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    """

    resources = {
        "train_images": (
            "emnist-digits-train-images-idx3-ubyte.gz",
            "d2662ecdc47895a6bbfce25de9e9a677",
        ),
        "train_labels": (
            "emnist-digits-train-labels-idx1-ubyte.gz",
            "2223fcfee618ac9c89ef20b6e48bcf9e",
        ),
        "test_images": (
            "emnist-digits-test-images-idx3-ubyte.gz",
            "a159b8b3bd6ab4ed4793c1cb71a2f5cc",
        ),
        "test_labels": (
            "emnist-digits-test-labels-idx1-ubyte.gz",
            "8afde66ea51d865689083ba6bb779fac",
        ),
    }


class Letters(_Split):
    """
    EMNIST Letters
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.
    _train_images, _train_labels, _test_images, _test_labels : np.ndarray, optional
        Numpy array representation of train/test images/labels from EMNIST Letters dataset.
        May be None if wasn't loaded manually or during initialization.
        If is not None, corresponding getter, e.g., _train_images -> train_images(),
        will be available.
    """

    resources = {
        "train_images": (
            "emnist-letters-train-images-idx3-ubyte.gz",
            "8795078f199c478165fe18db82625747",
        ),
        "train_labels": (
            "emnist-letters-train-labels-idx1-ubyte.gz",
            "c16de4f1848ddcdddd39ab65d2a7be52",
        ),
        "test_images": (
            "emnist-letters-test-images-idx3-ubyte.gz",
            "382093a19703f68edac6d01b8dfdfcad",
        ),
        "test_labels": (
            "emnist-letters-test-labels-idx1-ubyte.gz",
            "d4108920cd86601ec7689a97f2de7f59",
        ),
    }
