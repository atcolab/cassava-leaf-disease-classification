import torch
from tqdm import tqdm
import numpy as np
import config
from utils import rand_bbox
from torch.cuda.amp import autocast, GradScaler

def train_loop_fn(model, loader, optimizer, loss_func, device, epoch, scaler):
    model.train()

    TRAIN_LOSS = []

    bar = tqdm(enumerate(loader), total=len(loader))

    for step, (data, target) in bar:
        data = data.to(device)
        target = target.to(device)
        
        with autocast():
            
            rand_p = np.random.rand()

            if rand_p < 0.5:
                lam = np.random.beta(1., 1.)
                rand_index = torch.randperm(data.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
                outputs = model(data)
                loss = loss_func(outputs, target_a) * lam + loss_func(outputs, target_b) * (1. - lam)
            else:
                outputs = model(data)
                loss = loss_func(outputs, target)

            scaler.scale(loss).backward()

            TRAIN_LOSS.append(loss.item())
            smooth_loss = np.mean(TRAIN_LOSS[-30:])
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

            if ((step + 1) % config.ACCUM_ITER == 0) or ((step + 1) == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


        avg_train_loss = np.mean(TRAIN_LOSS)
    
    return avg_train_loss


def valid_loop_fn(model, loader, loss_func, device):
    model.eval()

    correct = 0.0
    total_samples = 0.0

    VAL_LOSS = []

    bar = tqdm(enumerate(loader), total=len(loader))

    with torch.no_grad():
        for step, (data, target) in bar:

            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)

            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size()[0]
            loss = loss_func(outputs, target)

            VAL_LOSS.append(loss.item())

            smooth_loss = np.mean(VAL_LOSS[-30:])
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    avg_valid_loss = np.mean(VAL_LOSS)
    accuracy = 100.0 * correct / total_samples

    return avg_valid_loss, accuracy