import torch
import numpy as np
from apex import amp
from tqdm import tqdm

def train_loop_fn(model, train_idx, loader, optimizer, loss_fn, device):
    model.train()

    avg_train_loss = 0.0
    train_running_acc = 0.0
    correct = 0.0

    losses = []
    accs = []

    for step, (data, target) in tqdm(enumerate(loader), total=len(loader)):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, target)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        avg_train_loss += loss.item() / len(loader)
        optimizer.step()

        correct += (outputs.argmax(1) == target).sum().item()
        train_running_acc = correct / len(train_idx)

    losses.append(avg_train_loss)
    accs.append(train_running_acc)

    return np.array(losses).mean(), np.array(accs).mean()

def val_loop_fn(model, valid_idx, loader, optimizer, loss_fn, device):
    model.eval()
    avg_val_loss = 0.0
    val_running_acc = 0.0
    correct = 0.0
    val_losses = []
    val_accs = []

    with torch.no_grad():

      for step, (data, target) in tqdm(enumerate(loader),total=len(loader)):

        data = data.to(device)
        target = target.to(device)

        outputs = model(data)
        loss = loss_fn(outputs, target.squeeze(-1))
        avg_val_loss += loss.item() / len(loader) 

        correct += (outputs.argmax(1) == target).sum().item()
        val_running_acc = correct / len(valid_idx)

    val_losses.append(avg_val_loss)
    val_accs.append(val_running_acc)

    return np.array(val_losses).mean(), np.array(val_accs).mean()






