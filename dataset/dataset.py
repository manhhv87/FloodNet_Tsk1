import torch
from skimage import io
from torch.utils.data import Dataset


class FloodNetTsk1(Dataset):
    """
    This is to use with array of files and array of labels as an input
    """

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = io.imread(self.X[idx])

        if self.transform is not None:
            image = self.transform(image)
        if self.y is None:
            return (image, idx)

        return (image, self.y[idx])
