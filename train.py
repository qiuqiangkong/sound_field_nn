from __future__ import annotations

import argparse
from pathlib import Path

import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from einops import rearrange
import museval

import wandb
from sound_field_prediction.data.samplers import InfiniteSampler
from sound_field_prediction.utils import LinearWarmUp, parse_yaml


def train(args) -> None:

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
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

    # LLM decoder
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    loss_fn = get_loss_fn(configs)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Logger
    if wandb_log:
        wandb.init(project="music_source_separation", name="{}".format(config_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        input_x = data['x'][:, 0 : -1, :, :].to(device)  # (b, t, w, h)
        target = data["x"][:, 1:, :, :].to(device)  # (b, t, w, h)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        output = model(input_x)

        # 2.3 Loss
        loss = loss_fn(output=output, target=target)
        
        # 2.4 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 2.5 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print(loss)
        
        '''
        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            train_sdr = validate(
                configs=configs,
                dataset=train_dataset, 
                model=model
            )

            test_sdr = validate(
                configs=configs,
                dataset=test_dataset, 
                model=model
            )

            if wandb_log:
                wandb.log(
                    data={"train_SDR": train_sdr, "test_SDR": test_sdr},
                    step=step
                )

            print("Train SDR: {}".format(train_sdr))
            print("Test SR: {}".format(test_sdr))
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break
        '''
        
        
        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    # from audidata.io.crops import RandomCrop, StartCrop
    # from audidata.transforms import Mono, TextNormalization, TimeShift

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    target_stem = configs["target_stem"]
    datasets_split = "{}_datasets".format(split)

    datasets = []
    
    for name in configs[datasets_split].keys():
    
        if name == "MUSDB18HQ":

            # from audidata.datasets import MUSDB18HQ

            # dataset = MUSDB18HQ(
            #     root=configs[datasets_split][name]["root"],
            #     split=configs[datasets_split][name]["split"],
            #     sr=sr,
            #     crop=RandomCrop(clip_duration=clip_duration, end_pad=0.),
            #     target_stems=[target_stem],
            #     time_align="group",
            #     mixture_transform=None,
            #     group_transform=None,
            #     stem_transform=None
            # )
            # datasets.append(dataset)

            dataset = PAWS()
            return dataset

        else:
            raise ValueError(name)


def get_pressure(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]
    data = data.reshape(-1,128,128).transpose(0,2,1)
    return data

    
class PAWS(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):

        x = get_pressure("/public/acoustic_field_data/training_set/complicated_room1_save1.npy")

        T, W, H = x.shape

        length = 10

        start_sample = random.randint(0, T - 10)

        data = {"x": x[start_sample : start_sample + length]}

        return data

    def __len__(self):
        return 1



def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize LLM decoder."""

    name = configs["model"]["name"]

    if name == "Cnn":

        from sound_field_prediction.models.cnn import Cnn

        return Cnn(config=None)

    elif name == "UNet":

        from sound_field_prediction.models.unet import UNet, UNetConfig

        config = UNetConfig(
            n_fft=configs["model"]["n_fft"],
            hop_length=configs["model"]["hop_length"],
        )

        model = UNet(config)

    else:
        raise ValueError(name)    

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


def get_loss_fn(configs):

    loss_type = configs["train"]["loss"]

    if loss_type == "l1":
        from sound_field_prediction.losses import l1_loss
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
    # valid_songs=10
    valid_songs = 10
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device
    losses = []

    clip_duration = configs["clip_duration"]
    sr = configs["sample_rate"]
    clip_samples = round(clip_duration * sr)
    batch_size = configs["train"]["batch_size_per_device"]
    target_stem = configs["target_stem"]

    skip_n = max(1, len(dataset) // valid_songs)

    all_sdrs = []

    for idx in range(0, len(dataset), skip_n):
        print("{}/{}".format(idx, len(dataset)))

        data = {}
        stems = ["vocals", "bass", "drums", "other"]

        for stem in stems:
            audio_path = dataset[idx]["{}_audio_path".format(stem)]
            audio, _ = librosa.load(
                audio_path,
                sr=configs["sample_rate"],
                mono=False
            )
            data[stem] = audio

        data["mixture"] = np.sum([data[stem] for stem in stems], axis=0)

        output = separate(
            model=model, 
            audio=data["mixture"], 
            clip_samples=clip_samples,
            batch_size=batch_size
        )
        target = data[target_stem]

        museval_sr = 44100
        output = librosa.resample(y=output, orig_sr=sr, target_sr=museval_sr)
        target = librosa.resample(y=target, orig_sr=sr, target_sr=museval_sr)

        # Calculate SDR. Shape should be (sources_num, channels_num, audio_samples)
        (sdrs, _, _, _) = museval.evaluate([target.T], [output.T])

        sdr = np.nanmedian(sdrs)
        all_sdrs.append(sdr)

    return np.nanmedian(all_sdrs)


def separate(
    model: nn.Module, 
    audio: np.ndarray, 
    clip_samples: int, 
    batch_size: int
) -> np.ndarray:
    r"""Separate a long audio.
    """

    device = next(model.parameters()).device

    audio_samples = audio.shape[1]
    padded_audio_samples = round(np.ceil(audio_samples / clip_samples) * clip_samples)
    audio = librosa.util.fix_length(data=audio, size=padded_audio_samples, axis=-1)

    clips = librosa.util.frame(
        audio, 
        frame_length=clip_samples, 
        hop_length=clip_samples
    )
    # shape: (c, t, clips_num)

    # clips = torch.Tensor(clips).to(device)

    # clips = rearrange(clips, 'c t n -> n c t')
    
    clips = clips.transpose(2, 0, 1)#.contiguous()
    # shape: (clips_num, c, t)

    clips = torch.Tensor(clips.copy()).to(device)

    clips_num = clips.shape[0]

    pointer = 0
    outputs = []

    while pointer < clips_num:

        batch_clips = torch.Tensor(clips[pointer : pointer + batch_size])

        with torch.no_grad():
            model.eval()
            batch_output = model(mixture=batch_clips)
            batch_output = batch_output.cpu().numpy()

        outputs.append(batch_output)
        pointer += batch_size

    outputs = np.concatenate(outputs, axis=0)
    # shape: (clips_num, channels_num, clip_samples)

    channels_num = outputs.shape[1]
    outputs = outputs.transpose(1, 0, 2).reshape(channels_num, -1)
    # shape: (channels_num, clips_num * clip_samples)

    outputs = outputs[:, 0 : audio_samples]
    # shape: (channels_num, audio_samples)

    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)