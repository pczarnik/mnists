from typing import Optional

from .dataset import SplitDataset, ZippedDataset


class EMNIST(SplitDataset):
    """
    EMNIST Dataset
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    Balanced, ByClass, ByMerge, Digits, Letters : class
        Child classes containing splits of EMNIST dataset.

    Usage
    -----
    >>> from mnists import EMNIST
    >>> emnist = EMNIST()
    >>> letters = emnist.Letters()
    >>> letters.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @article{cohen2017emnist,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
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
        super().__init__(target_dir, download, force_download)

        self.Balanced = self._create_split(Balanced)
        self.ByClass = self._create_split(ByClass)
        self.ByMerge = self._create_split(ByMerge)
        self.Digits = self._create_split(Digits)
        self.Letters = self._create_split(Letters)


class Balanced(ZippedDataset):
    """
    EMNIST Balanced
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.

    Usage
    -----
    >>> from mnists import EMNIST
    >>> emnist = EMNIST()
    >>> balanced = emnist.Balanced()
    >>> balanced.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @article{cohen2017emnist,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
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


class ByClass(ZippedDataset):
    """
    EMNIST ByClass
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.

    Usage
    -----
    >>> from mnists import EMNIST
    >>> emnist = EMNIST()
    >>> byclass = emnist.ByClass()
    >>> byclass.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @article{cohen2017emnist,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
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


class ByMerge(ZippedDataset):
    """
    EMNIST ByMerge
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.

    Usage
    -----
    >>> from mnists import EMNIST
    >>> emnist = EMNIST()
    >>> bymerge = emnist.ByMerge()
    >>> bymerge.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @article{cohen2017emnist,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
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


class Digits(ZippedDataset):
    """
    EMNIST Digits
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.

    Usage
    -----
    >>> from mnists import EMNIST
    >>> emnist = EMNIST()
    >>> digits = emnist.Digits()
    >>> digits.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @article{cohen2017emnist,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
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


class Letters(ZippedDataset):
    """
    EMNIST Letters
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    target_dir : str
        Directory where all files exist or will be downloaded.
    zip_filepath : str
        Zip file from which dataset will be extracted.

    Usage
    -----
    >>> from mnists import EMNIST
    >>> emnist = EMNIST()
    >>> letters = emnist.Letters()
    >>> letters.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @article{cohen2017emnist,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
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
