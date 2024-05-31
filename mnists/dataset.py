import os
import tempfile
from typing import Optional

import numpy as np

from .utils import check_file_integrity, download_file, extract_from_zip, read_idx_file

TEMPORARY_DIR = os.path.join(tempfile.gettempdir(), "mnists")


class Dataset:
    mirrors = []
    resources = {}

    def __init__(
        self,
        target_dir: Optional[str] = None,
        download: bool = True,
        force_download: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        target_dir : str, default='/tmp/<dataset_name>/'
            Directory where all files exist or will be downloaded to (if `download` is True).
        download : bool, default=True
            If True and files don't exist in `target_dir`, downloads all files to `target_dir`.
        force_download : bool, default=False
            If True, downloads all files to `target_dir`, even if they exist there.
        """

        self.target_dir = (
            os.path.join(TEMPORARY_DIR, type(self).__name__)
            if target_dir is None
            else target_dir
        )

        if download or force_download:
            self.download(force_download)

    def download(self, force: bool = False) -> None:
        """
        Download files from mirrors and save to `target_dir`.

        Parameters
        ----------
        force : bool=False
            If True, downloads all files even if they exist.
        """

        os.makedirs(self.target_dir, exist_ok=True)

        for filename, md5 in self.resources.values():
            filepath = os.path.join(self.target_dir, filename)

            if not force and check_file_integrity(filepath, md5):
                continue

            download_file(self.mirrors, filename, filepath)


class IdxDataset(Dataset):
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
        target_dir : str, default='/tmp/<dataset_name>/'
            Directory where all files exist or will be downloaded to (if `download` is True).
        download : bool, default=True
            If True and files don't exist in `target_dir`, downloads all files to `target_dir`.
        force_download : bool, default=False
            If True, downloads all files to `target_dir`, even if they exist there.
        load : bool, default=True
            If True, loads data from files in `target_dir`.
        """

        self.target_dir = (
            os.path.join(TEMPORARY_DIR, type(self).__name__)
            if target_dir is None
            else target_dir
        )

        self._train_images: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._test_images: Optional[np.ndarray] = None
        self._test_labels: Optional[np.ndarray] = None

        if download or force_download:
            self.download(force_download)

        if load:
            self.load()

    def train_images(self) -> np.ndarray:
        """
        Return train_images numpy array.

        Returns
        -------
        np.ndarray
        """
        if self._train_images is None:
            self._raise_dataset_not_loaded()
        return self._train_images

    def train_labels(self) -> np.ndarray:
        """
        Return train_labels numpy array.

        Returns
        -------
        np.ndarray
        """
        if self._train_labels is None:
            self._raise_dataset_not_loaded()
        return self._train_labels

    def test_images(self) -> np.ndarray:
        """
        Return test_images numpy array.

        Returns
        -------
        np.ndarray
        """
        if self._test_images is None:
            self._raise_dataset_not_loaded()
        return self._test_images

    def test_labels(self) -> np.ndarray:
        """
        Return test_labels numpy array.

        Returns
        -------
        np.ndarray
        """
        if self._test_labels is None:
            self._raise_dataset_not_loaded()
        return self._test_labels

    def _raise_dataset_not_loaded(self):
        raise RuntimeError(
            "Dataset wasn't loaded. You need to run .load() or create new "
            "object with load=True"
        )

    def load(self, transpose=False) -> None:
        """
        Load data from files in `target_dir`.

        Parameters
        ----------
        transpose : bool=False
            If True, transposes train and test images.
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
            setattr(self, f"_{key}", data)

        if transpose:
            self._transpose_images()

    def _transpose_images(self) -> None:
        self._train_images = np.moveaxis(self._train_images, -2, -1)
        self._test_images = np.moveaxis(self._test_images, -2, -1)


class SplitDataset(Dataset):
    resources = {"gzip": (None, None)}

    def _create_split(self, split_cls: type["ZippedDataset"]) -> type["ZippedDataset"]:
        split_cls.default_base_dir = self.target_dir
        file, md5 = self.resources["gzip"]
        split_cls.default_zip_filepath = os.path.join(self.target_dir, file)
        split_cls.zip_md5 = md5
        return split_cls


class ZippedDataset(IdxDataset):
    default_base_dir = None
    default_zip_filepath = None
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
