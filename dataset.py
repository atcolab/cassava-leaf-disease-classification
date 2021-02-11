import cv2
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset

import config

class LeafDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fname = self.df['image_id'].values[index]
        fpath = f'{config.TRAIN_IMG_PATH}{fname}'

        image = cv2.imread(fpath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)
            image = image['image']

        label = self.labels[index]

        return torch.tensor(image), torch.tensor(label)