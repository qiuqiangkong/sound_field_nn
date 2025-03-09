import torch


def l1_loss(output, target):
	return torch.mean(torch.abs(output - target))