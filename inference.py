from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from sound_field_nn.datasets.fdtd_2d import FDTD2D
from sound_field_nn.utils import parse_yaml
from train import get_model


def inference(args):

    # Arguments and parameters
    config_path = args.config
    ckpt_path = args.ckpt_path
    device = "cuda"

    # Default parameters
    configs = parse_yaml(config_path)
    skip = configs["test_datasets"]["FDTD2D"]["skip"]
    
    # Load model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)

    # Data simulator
    simulator = FDTD2D()
    data = simulator()

    # Input data
    bnd = data["bnd"]  # (h, w)
    u = data["u"][0 :: skip]  # (t, h, w)
    u0 = torch.Tensor(u[0 : 1, None, :, :]).to(device)  # (b, c, h, w)
    bnd = torch.Tensor(bnd[None, None, :, :]).to(device)  # (b, c, h, w)
    x = torch.cat((u0, bnd), dim=1)  # (b, c, h, w)

    # Outputs
    outputs = [u0]

    N = 5  # steps to run
    with torch.no_grad():
        for _ in range(N):
            model.eval()
            u_next = model(x)  # (b, c, h, w)
            outputs.append(u_next)
            x = torch.cat((u_next, bnd), dim=1)  # (b, c, h, w)

    # Visualization
    fig, axes = plt.subplots(2, N, sharex=True)
    for n in range(N):
        axes[0, n].matshow(outputs[n][0, 0].data.cpu().numpy(), origin='lower', aspect='equal', cmap='jet', vmin=-1, vmax=1)
        axes[1, n].matshow(u[n], origin='lower', aspect='equal', cmap='jet', vmin=-1, vmax=1)
        axes[0, n].set_title("Pred")
        axes[1, n].set_title("GT")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    plt.tight_layout()

    out_path = "pred.pdf"
    plt.savefig(out_path)
    print(f"Write out to {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--ckpt_path", type=str, required=True, default=False)
    args = parser.parse_args()

    inference(args)