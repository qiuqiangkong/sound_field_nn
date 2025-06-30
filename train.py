from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from sound_field_nn.losses import l1_loss
from sound_field_nn.samplers.infinite_sampler import InfiniteSampler
from sound_field_nn.utils import LinearWarmUp, parse_yaml


def train(args) -> None:

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Checkpoints directory
    ckpts_dir = Path("./checkpoints", filename, Path(config_path).stem)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train")
    test_dataset = get_dataset(configs, split="test")

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    # Loss function
    loss_fn = get_loss_fn(configs)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Logger
    if wandb_log:
        wandb.init(project="sound_field_nn", name=Path(config_path).stem)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):
        
        # ------ 1. Data preparation ------
        # Input
        curr_u = data["curr_u"][:, None, :, :].to(device)
        bnd = data["bnd"][:, None, :, :].to(device)
        x = torch.cat((curr_u, bnd), dim=1)  # (b, c, h, w)
        
        # Target
        target = data["next_u"][:, None, :, :].to(device)  # (b, c, h, w)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        output = model(x)

        # 2.2 Loss
        loss = loss_fn(output=output, target=target)
        
        # 2.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 2.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print(loss)

        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            test_loss = validate(
                configs=configs,
                dataset=test_dataset, 
                model=model
            )

            if wandb_log:
                wandb.log(
                    data={"test_loss": test_loss},
                    step=step
                )

            print("Test loss: {:.4f}".format(test_loss))
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break
        
        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    ds = f"{split}_datasets"
    for name in configs[ds].keys():
    
        if name == "FDTD2D":
            from sound_field_nn.datasets.fdtd_2d import FDTD2D_Slice
            dataset = FDTD2D_Slice(configs[ds][name]["skip"])
            return dataset

        else:
            raise ValueError(name)


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""
    
    name = configs["model"]["name"]

    if name == "Cnn":
        from sound_field_nn.models.cnn import Cnn
        model = Cnn()

    else:
        raise ValueError(name)    

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
    
    return model


def get_loss_fn(configs: dict) -> callable:

    loss_type = configs["train"]["loss"]

    if loss_type == "l1":
        from sound_field_nn.losses import l1_loss
        return l1_loss

    else:
        raise ValueError(loss_type)


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler
        

def validate(
    configs: dict,
    dataset: Dataset,
    model: nn.Module,
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device
    losses = []
    valid_num = 20

    for idx in range(valid_num):

        # ------ 1. Data preparation ------
        data = dataset[idx]
        data = default_collate([data])

        # Input
        curr_u = data["curr_u"][:, None, :, :].to(device)
        bnd = data["bnd"][:, None, :, :].to(device)
        x = torch.cat((curr_u, bnd), dim=1)  # (b, c, h, w)

        # Target
        target = data["next_u"][:, None, :, :].to(device)  # (b, c, h, w)
        
        # ------ 2. Evaluation ------
        # 2.1 Forward
        with torch.no_grad():
            model.eval()
            output = model(x)

        # 2.2 Loss
        loss = l1_loss(output=output, target=target)
        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)