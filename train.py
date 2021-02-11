import os
import sys
import time
import math

from model import resnet34
import config
import engine
import dataset
import transforms

import numpy as np
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensor

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from apex import amp
import pretrainedmodels
from torchcontrib.optim import SWA
# from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm.notebook import tqdm
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

def run():
    df = pd.read_csv(f"{config.ROOT}/train.csv")

    folds = df.copy()
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(folds)):

        train_test = folds.iloc[train_idx]
        train_test.reset_index(drop=True, inplace=True)  

        valid_test = folds.iloc[valid_idx]
        valid_test.reset_index(drop=True, inplace=True)

        train_dataset = dataset.LeafDataset(
            train_test,
            train_test['label'],
            transforms.transforms_train
        )

        valid_dataset = dataset.LeafDataset(
            valid_test,
            valid_test['label'],
            transforms.transforms_valid
        )

        train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS, sampler=RandomSampler(train_dataset))
        valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=config.NUM_WORKERS, sampler=SequentialSampler(valid_dataset))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = resnet34().to(device)
        optimizer = Adam(model.parameters(), lr=config.LR)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        scheduler = CosineAnnealingLR(optimizer, config.NUM_EPOCHS)

        loss_fn = nn.CrossEntropyLoss().to(device)

        num_epochs=config.NUM_EPOCHS
        kernel_type = config.KERNEL_TYPE
        best_file = f'{kernel_type}_best_fold{fold}.bin'
        acc_max = 0

        loss_history = {
            "train": [],
            "valid": []
        }

        acc_history = {
            "train": [],
            "valid": []
        }

        for epoch in range(num_epochs):
            scheduler.step(epoch)
            train_losses, train_acc = engine.train_loop_fn(model, train_idx, train_loader, optimizer, loss_fn, device)

            loss_history['train'].append(train_losses)
            acc_history['train'].append(train_acc)

            val_losses, val_acc = engine.val_loop_fn(model, valid_idx, valid_loader, optimizer, loss_fn, device)

            loss_history['valid'].append(val_losses)
            acc_history['valid'].append(val_acc)

            print(f"Epoch: {epoch + 1} | lr: {optimizer.param_groups[0]['lr']:.7f} | train loss: {train_losses:.4f} | val loss: {val_losses:.4f} | train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

            if val_acc > acc_max:
                print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(acc_max, val_acc))
                torch.save(model.state_dict(), best_file)
                acc_max = val_acc


if __name__ == "__main__":
    run()
