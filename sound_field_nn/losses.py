import torch
from torch import Tensor


def l1_loss(output: Tensor, target: Tensor) -> torch.float:
	return (output - target).abs().mean()