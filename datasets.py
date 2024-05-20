import torch
import cv2

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import glob

class ASLTrainDataset(Dataset):
    def __init__(self, dir: str):
        super().__init__()
        self.dataset = ImageFolder(dir, transform=ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
    
    @property
    def classes(self):
        return self.dataset.classes

class ASLTestDataset(Dataset):

    test_dataset = []

    def __init__(self):
        super().__init__()
        for i, item in enumerate(glob.glob('./dataset/asl_alphabet_test/*')):
            dir, filename = item.split("\\")
            image = torch.tensor(cv2.imread(f"{dir}/{filename}")) / 255
            image = torch.moveaxis(image, -1, 0)
            self.test_dataset.append((image, i))

    def __len__(self):
        return len(self.test_dataset)
    
    def __getitem__(self, idx):
        return self.test_dataset[idx]