import torch

def jaccard_index(pred, target, smooth=1e-6):
    pred = pred > 0.5
    target = target > 0.5
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + smooth) / (union + smooth)

def dice_score(pred, target, smooth=1e-6):
    pred = pred > 0.5
    target = target > 0.5
    intersection = (pred & target).float().sum()
    return (2. * intersection + smooth) / (pred.float().sum() + target.float().sum() + smooth)