import torch
import torch.nn.functional as F
from math import exp

def l1_loss(network_output, gt, mask=None):
    assert network_output.shape == gt.shape
    if mask is not None:
        if gt.shape[-2:] != mask.shape:
            mask = F.interpolate(mask[None, None].half(), size=gt.shape[-2:], mode='nearest')[0, 0].bool()
        gt = gt[..., mask > 0]
        network_output = network_output[..., mask > 0]
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt, mask=None):
    assert network_output.shape == gt.shape
    if mask is not None:
        if gt.shape[-2:] != mask.shape:
            mask = F.interpolate(mask[None, None].half(), size=gt.shape[-2:], mode='nearest')[0, 0].bool()
        gt[..., mask == 0] = 0
        network_output[..., mask == 0] = 0
    return ((network_output - gt) ** 2).mean()

def cosine_loss(network_output, gt, mask=None):
    assert network_output.shape == gt.shape
    if mask is not None:
        if gt.shape[-2:] != mask.shape:
            mask = F.interpolate(mask[None, None].half(), size=gt.shape[-2:], mode='nearest')[0, 0].bool()
        gt = gt[..., mask > 0]
        network_output = network_output[..., mask > 0]
    return (1 - F.cosine_similarity(network_output, gt, dim=0)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
