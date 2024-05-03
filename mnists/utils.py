import gzip
import hashlib
import os
import struct

import numpy as np

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
