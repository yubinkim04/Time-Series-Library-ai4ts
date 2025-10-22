import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


# def adjust_learning_rate(optimizer, epoch, args):
#     # lr = args.learning_rate * (0.2 ** (epoch // 2))
#     if args.lradj == 'type1':
#         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
#     elif args.lradj == 'type2':
#         lr_adjust = {
#             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
#             10: 5e-7, 15: 1e-7, 20: 5e-8
#         }
#     elif args.lradj == 'type3':
#         lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
#     elif args.lradj == "cosine":
#         lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
#     elif args.lradj == "warmup":
#         # ✅ Fixed: proper linear warmup, then cosine decay (great for Transformers)
#         warmup_epochs = getattr(args, "warmup_epochs", max(1, int(0.1 * args.train_epochs)))
#         min_lr = getattr(args, "min_lr", 0.0)

#         if epoch <= warmup_epochs:
#             # linear ramp: 0 → base_lr over warmup_epochs
#             lr = args.learning_rate * (epoch / float(warmup_epochs))
#         else:
#             # cosine decay from base_lr → min_lr over the remaining epochs
#             T = max(1, args.train_epochs - warmup_epochs)
#             t = min(T, epoch - warmup_epochs)
#             cosine = 0.5 * (1 + math.cos(math.pi * t / float(T)))
#             lr = min_lr + (args.learning_rate - min_lr) * cosine
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))

def adjust_learning_rate(optimizer, epoch, args):
    """
    Updates optimizer.param_groups[i]['lr'] in-place.

    Schedules supported:
      - type1: step decay by 0.5 each epoch (from epoch 1)
      - type2: fixed epoch -> lr mapping
      - type3: hold for 2 epochs, then decay by 0.9 each epoch
      - cosine: cosine over total train epochs (no warmup)
      - warmup: linear warmup, then cosine decay to min_lr

    Notes:
      - Assumes epoch is 1-based.
      - args should have: learning_rate, train_epochs (or epochs), and optionally warmup_epochs, min_lr.
    """
    base_lr = args.learning_rate
    total_epochs = getattr(args, "train_epochs", getattr(args, "epochs", None))
    if total_epochs is None:
        raise ValueError("Please provide args.train_epochs (or args.epochs).")

    lr = None

    if args.lradj == 'type1':
        # decay by 0.5 each epoch starting at epoch 1
        lr = base_lr * (0.5 ** (epoch - 1))

    elif args.lradj == 'type2':
        mapping = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        lr = mapping.get(epoch, None)  # only update on listed epochs

    elif args.lradj == 'type3':
        lr = base_lr if epoch < 3 else base_lr * (0.9 ** (epoch - 3))

    elif args.lradj == "cosine":
        # pure cosine from base_lr to 0 across all epochs
        progress = min(1.0, max(0.0, epoch / float(total_epochs)))
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * progress))

    elif args.lradj == "warmup":
        # ✅ Fixed: proper linear warmup, then cosine decay (great for Transformers)
        warmup_epochs = getattr(args, "warmup_epochs", max(1, int(0.1 * total_epochs)))
        min_lr = getattr(args, "min_lr", 0.0)

        if epoch <= warmup_epochs:
            # linear ramp: 0 → base_lr over warmup_epochs
            lr = base_lr * (epoch / float(warmup_epochs))
        else:
            # cosine decay from base_lr → min_lr over the remaining epochs
            T = max(1, total_epochs - warmup_epochs)
            t = min(T, epoch - warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * t / float(T)))
            lr = min_lr + (base_lr - min_lr) * cosine

    # Apply if we computed a new LR this epoch
    if lr is not None:
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        print('Updating learning rate to {:.6g}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
