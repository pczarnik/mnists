import array
import functools
import gzip
import hashlib
import os
import struct
import tempfile
from typing import Optional
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlretrieve

import numpy as np

TEMPORARY_DIR = tempfile.gettempdir()


class MNIST:
    mirrors = [
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

    default_dir = os.path.join(TEMPORARY_DIR, "mnist")

    def __init__(
        self,
        target_dir: Optional[str] = None,
        download: bool = True,
        force_download: bool = False,
        load: bool = True,
    ) -> None:
        self.target_dir = self.default_dir if target_dir is None else target_dir

        self.train_images: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_images: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None

        if download:
            self.download(force_download)

        if load:
            self.load()

    def download(self, force: bool = False) -> None:
        os.makedirs(self.target_dir, exist_ok=True)

        for filename, md5 in self.resources.values():
            filepath = os.path.join(self.target_dir, filename)

            if not force and self._exists(filepath, md5):
                continue

            self._download_file(filename, filepath)

    def load(self) -> None:
        for key, (filename, md5) in self.resources.items():
            filepath = os.path.join(self.target_dir, filename)

            if not self._exists(filepath, md5):
                raise RuntimeError(
                    "Dataset not found. Use download=True or mnist.download() to download it"
                )

            fopen = gzip.open if os.path.splitext(filepath)[1] == ".gz" else open

            with fopen(filepath, "rb") as fd:
                parsed = self._parse_idx(fd)
                setattr(self, key, parsed)

    def _exists(self, filepath: str, md5: str) -> bool:
        return os.path.isfile(filepath) and md5 == self._calculate_md5(filepath)

    def _calculate_md5(self, filepath: str, chunk_size: int = 1024 * 1024) -> str:
        md5 = hashlib.md5()
        with open(filepath, "rb") as fd:
            while chunk := fd.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

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

    def _parse_idx(self, fd) -> np.ndarray:
        DATA_TYPES = {
            0x08: "B",  # unsigned byte
            0x09: "b",  # signed byte
            0x0B: "h",  # short (2 bytes)
            0x0C: "i",  # int (4 bytes)
            0x0D: "f",  # float (4 bytes)
            0x0E: "d",  # double (8 bytes)
        }

        header = fd.read(4)
        if len(header) != 4:
            raise RuntimeError(
                "Invalid IDX file, file empty or does not contain a full header."
            )

        zeros, data_type, num_dimensions = struct.unpack(">HBB", header)

        if zeros != 0:
            raise RuntimeError(
                "Invalid IDX file, file must start with two zero bytes. "
                "Found 0x%02x" % zeros
            )

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise RuntimeError("Unknown data type 0x%02x in IDX file" % data_type)

        dimension_sizes = struct.unpack(
            ">" + "I" * num_dimensions, fd.read(4 * num_dimensions)
        )

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(lambda x, y: x * y, dimension_sizes)
        if len(data) != expected_items:
            raise RuntimeError(
                "IDX file has wrong number of items. "
                "Expected: %d. Found: %d" % (expected_items, len(data))
            )

        return np.array(data).reshape(dimension_sizes)
