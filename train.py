import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import config
from dataset import LeafDataset
from model import ViT
from engine import train_loop_fn, valid_loop_fn
from transforms import transforms_train, transforms_valid
from loss import LabelSmoothingCrossEntropy
from scheduler import GradualWarmupSchedulerV2

import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

import timm
from pprint import pprint
from torchcontrib.optim import SWA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from torch.utils.data.sampler import (
    SubsetRandomSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)


def run():
    print(torch.__version__)

    df = pd.read_csv(config.FOLDS_CSV_PATH)
    print(df.shape)
    print(df.head())

    num_gpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ViT().to(device)
    # model = convert_model(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LR / config.WARMUP_FACTOR)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.N_EPOCHS - config.WARMUP_EPO
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer,
        multiplier=config.WARMUP_FACTOR,
        total_epoch=config.WARMUP_EPO,
        after_scheduler=scheduler_cosine,
    )

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids = range(num_gpu)).to(device)

    # scheduler = CosineAnnealingLR(optimizer, config.N_EPOCHS)
    loss_func = LabelSmoothingCrossEntropy()

    folds = df.copy()

    for fold in range(config.N_FOLDS):

        print(f"Fold: {fold+1} / {config.N_FOLDS}")

        train_idx = np.where((folds["fold"] != fold))[0]
        valid_idx = np.where((folds["fold"] == fold))[0]

        df_this = folds.loc[train_idx].reset_index(drop=True)
        df_valid = folds.loc[valid_idx].reset_index(drop=True)

        dataset_train = LeafDataset(
            df_this, df_this["label"], transform=transforms_train
        )
        dataset_valid = LeafDataset(
            df_valid, df_valid["label"], transform=transforms_valid
        )

        train_loader = DataLoader(
            dataset_train,
            batch_size=config.TRAIN_BS,
            num_workers=config.NUM_WORKERS,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset_valid,
            batch_size=config.VALID_BS,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )

        kernel_type = "vit"
        best_file = f"{kernel_type}_best_fold{fold}.bin"
        acc_max = 0

        for epoch in range(1, config.N_EPOCHS + 1):

            scaler = GradScaler()
            scheduler.step(epoch - 1)
            avg_train_loss = train_loop_fn(
                model, train_loader, optimizer, loss_func, device, epoch, scaler
            )
            avg_valid_loss, accuracy = valid_loop_fn(
                model, valid_loader, loss_func, device
            )

            content = f"Epoch: {epoch} | lr: {optimizer.param_groups[0]['lr']:.7f} | train loss: {avg_train_loss:.4f} | val loss: {avg_valid_loss:.4f} | accuracy: {accuracy:.4f}"
            print(content)

            with open(f"log_{kernel_type}.txt", "a") as appender:
                appender.write(content + "\n")

            if accuracy > acc_max:
                print(
                    "score2 ({:.6f} --> {:.6f}).  Saving model ...".format(
                        acc_max, accuracy
                    )
                )
                torch.save(model.state_dict(), best_file)
                acc_max = accuracy

            torch.save(model.state_dict(), f"{kernel_type}_final_fold.bin")


def eval_oof():

    df = pd.read_csv(config.FOLDS_CSV_PATH)

    folds = df.copy()
    oof = np.zeros((len(folds), 1))

    for fold in range(config.N_FOLDS):
        print(f"Fold: {fold+1} / {config.N_FOLDS}")

        train_idx = np.where((folds["fold"] != fold))[0]
        valid_idx = np.where((folds["fold"] == fold))[0]

        df_this = folds.loc[train_idx].reset_index(drop=True)
        df_valid = folds.loc[valid_idx].reset_index(drop=True)

        dataset_train = LeafDataset(
            df_this, df_this["label"], transform=transforms_train
        )
        dataset_valid = LeafDataset(
            df_valid, df_valid["label"], transform=transforms_valid
        )

        train_loader = DataLoader(
            dataset_train,
            batch_size=config.TRAIN_BS,
            num_workers=config.NUM_WORKERS,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset_valid,
            batch_size=config.VALID_BS,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        kernel_type = "vit"
        model = ViT().to(device)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model, device_ids = range(num_gpu)).to(device)

        model.load_state_dict(torch.load(f"{kernel_type}_best_fold{fold}.bin"))
        model.eval()  # switch model to the evaluation mode

        val_preds = torch.zeros((len(valid_idx), 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            for step, (data, target) in tqdm(enumerate(valid_loader)):
                data = data.to(device)
                target = target.to(device)

                outputs = model(data)

                pred = outputs.max(1, keepdim=True)[1]

                val_preds[step * 32 : step * 32 + 32] = pred
            oof[valid_idx] = val_preds.cpu().numpy()

    print("OOF: {:.3f}".format(accuracy_score(folds["label"], oof)))


if __name__ == "__main__":
    run()
    # eval_oof()