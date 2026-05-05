import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = Path(fdir)
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def _load_batch(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads one CIFAR-10 batch file.

        Returns:
            images: numpy array with shape (10000, 32, 32, 3), dtype uint8
            labels: numpy array with shape (10000,), dtype int64
        """
        if not file_path.is_file():
            raise ValueError(f"Missing CIFAR-10 file: {file_path}")

        with open(file_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

        data = batch[b"data"]
        labels = batch[b"labels"]

        images = data.reshape(-1, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)
        images = images.astype(np.uint8)

        labels = np.array(labels, dtype=np.int64)

        return images, labels

    def load_cifar(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        Labels should be returned either as a Python list of ints or as a
        numpy array with dtype int64.
        """
        if not self.fdir.is_dir():
            raise ValueError(f"CIFAR-10 directory does not exist: {self.fdir}")

        if self.subset == Subset.TRAINING:
            batch_files = [
                self.fdir / "data_batch_1",
                self.fdir / "data_batch_2",
                self.fdir / "data_batch_3",
                self.fdir / "data_batch_4",
            ]
        elif self.subset == Subset.VALIDATION:
            batch_files = [self.fdir / "data_batch_5"]
        elif self.subset == Subset.TEST:
            batch_files = [self.fdir / "test_batch"]
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

        images_list = []
        labels_list = []

        for batch_file in batch_files:
            images, labels = self._load_batch(batch_file)
            images_list.append(images)
            labels_list.append(labels)

        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0).astype(np.int64)

        return images, labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of length {len(self)}"
            )

        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.classes)