import gzip
import hashlib
import os
import struct
import time
import zipfile
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlretrieve

import numpy as np

_TQDM_ACTIVE = True
try:
    from tqdm import tqdm
except ImportError:
    tqdm = object
    _TQDM_ACTIVE = False


IDX_TYPEMAP = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.int16,
    0x0C: np.int32,
    0x0D: np.float32,
    0x0E: np.float64,
}


def read_idx_file(filepath: str) -> np.ndarray:
    """
    Read file in IDX format and return numpy array.

    Parameters
    ----------
    filepath : str
        Path to a IDX file. The file can be gzipped.

    Returns
    -------
    np.ndarray
        Data read from IDX file in numpy array.
    """

    fopen = gzip.open if os.path.splitext(filepath)[1] == ".gz" else open

    with fopen(filepath, "rb") as f:
        data = f.read()

    h_len = 4
    header = data[:h_len]
    zeros, dtype, ndims = struct.unpack(">HBB", header)

    if zeros != 0:
        raise RuntimeError(
            "Invalid IDX file, file must start with two zero bytes. "
            f"Found 0x{zeros:X}"
        )

    try:
        dtype = IDX_TYPEMAP[dtype]
    except KeyError as e:
        raise RuntimeError(f"Unknown data type 0x{dtype:02X} in IDX file") from e

    dim_offset = h_len
    dim_len = 4 * ndims
    dim_sizes = data[dim_offset : dim_offset + dim_len]
    dim_sizes = struct.unpack(">" + "I" * ndims, dim_sizes)

    data_offset = h_len + dim_len
    parsed = np.frombuffer(data, dtype=dtype, offset=data_offset)

    if parsed.shape[0] != np.prod(dim_sizes):
        raise RuntimeError(
            f"Declared size {dim_sizes}={np.prod(dim_sizes)} and "
            f"actual size {parsed.shape[0]} of data in IDX file don't match"
        )

    return parsed.reshape(dim_sizes)


def check_file_integrity(filepath: str, md5: str) -> bool:
    """
    Check if file exists and if exists if its MD5 checksum is correct.

    Parameters
    ----------
    filepath : str
        Path to a file.
    md5 : str
        Correct MD5 checksum of the file.

    Returns
    -------
    bool
        Returns True when file exists and its MD5 checksum is equal `md5`.
    """

    return os.path.isfile(filepath) and md5 == calculate_md5(filepath)


def calculate_md5(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate MD5 checksum of the file.

    Parameters
    ----------
    filepath : str
        Path to a file.
    chunk_size : int, default=1024 * 1024
        Size of chunks which will be read from the file.

    Returns
    -------
    str
        MD5 checksum of the file.
    """

    md5 = hashlib.md5()
    with open(filepath, "rb") as fd:
        while chunk := fd.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


class EmptyTqdm(object):
    # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/utils/tqdm_utils.py#L56
    def __init__(self, *args, **kwargs):
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        def empty_fn(*args, **kwargs):
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


class Tqdm(tqdm):
    # https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)


def custom_tqdm(*args, verbose, **kwargs):
    if _TQDM_ACTIVE and verbose:
        return Tqdm(*args, **kwargs)
    else:
        return EmptyTqdm(*args, **kwargs)


def download_file(
    mirrors: list[str],
    filename: str,
    filepath: str,
    verbose: bool = False,
) -> None:
    """
    Download file trying every mirror if the previous one fails.

    Parameters
    ----------
    mirrors : list[str]
        List of the URLs of the mirrors.
    filename: str
        Name of the file on the server.
    filepath : str
        Path to the output file.
    verbose : bool, default=False
        If True, prints download logs.
    """

    for mirror in mirrors:
        url = urljoin(mirror, filename)
        try:
            if verbose:
                print(f"Downloading {url} to {filepath}")
            with custom_tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=filepath,
                verbose=verbose,
            ) as t:
                urlretrieve(url, filepath, reporthook=t.update_to)
                t.total = t.n
            return
        except URLError as error:
            if verbose:
                print(f"Failed to download {url} (trying next mirror):\n{error}")
            continue

    raise RuntimeError(f"Error downloading {filename}")


def extract_from_zip(zip_path: str, filename: str, output_dir: str) -> None:
    """
    Extract file from zip and save it to given directory (with correct metadata).

    Parameters
    ----------
    zip_path : str
        Path to the zip archive.
    filename : str
        Name of the file to be extracted.
    output_dir : str
        Directory where the file will be saved.
    """

    with zipfile.ZipFile(zip_path, "r") as archive:
        file = list(
            filter(
                lambda s: os.path.basename(s.filename) == filename, archive.infolist()
            )
        )

        if len(file) != 1:
            raise RuntimeError(
                f"Error while extracting {filename}: "
                f"found {len(file)} corresponding files in {zip_path}"
            )

        file = file[0]

        file.filename = os.path.basename(file.filename)
        archive.extract(file, output_dir)

        # add correct datetime metadata
        date_time = time.mktime(file.date_time + (0, 0, -1))
        os.utime(os.path.join(output_dir, file.filename), (date_time, date_time))
