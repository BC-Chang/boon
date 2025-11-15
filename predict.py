import sys
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, GradientAccumulationScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functools import reduce, partial
from timeit import default_timer

from src.utils.utils import *
from src.models.base import FNO3d
from src.models.multi_step import BOON_FNO3d
from src.models.lightning.BOON_3d import BOON3D
from dataloader import get_dataloader, split_indices
import wandb
from lightning.pytorch.loggers import WandbLogger
import json

if __name__ == "__main__":

    # Preprocessing args
    torch.set_float32_matmul_precision("medium")
    net_name = "BOON_v0.0.10"
    
    datapath = Path(r"/scratch/06898/bchang/elec_sims")
    model = BOON3D
    phase = "test"
    with open(f'lightning_logs/{net_name}/hparam_config.json', 'r') as f:
        json_string = f.read()

    hparams = json.loads(json_string)
    hparams['model'] = model

    test_dataloader = get_dataloader(
        hparams[f'{phase}_ids'],
        image_size=(256, 256, 256),
        data_path=datapath,
        phase=phase,
        num_workers=71,
        persistent_workers=False,
    )
    
    try:
        model_dir = f"lightning_logs/{hparams['net_name']}/checkpoints"
        # model_loc = r"C:\Users\bcc2459\OneDrive - The University of Texas at Austin\Documents\PE-FNO3D\lightning_logs\PEBCNO-v0.4\checkpoints\epoch=39-val_loss=0.00-v1.ckpt"
        model_loc = glob.glob(f"{model_dir}/*val*.ckpt")[0]
        print(f"Loading {model_loc}")
        model = BOON3D.load_from_checkpoint(model_loc,
                                           model=hparams['model'],
                                           in_channels=1,
                                           out_channels=1,
                                           modes1=hparams['modes1'],
                                           modes2=hparams['modes2'],
                                           modes3=hparams['modes3'],
                                           width=hparams['width'],
                                           num_layers=hparams['num_layers'],
                                           lr=hparams['learning_rate'],)
    except IndexError:
        raise FileNotFoundError(
            f"Could not find checkpoint in {model_dir} or directory does not exist.")

    trainer = pl.Trainer()

    predictions = trainer.predict(model, dataloaders=test_dataloader)
    save_path = Path(f"/scratch/06898/bchang/BOON-FNO/{hparams['net_name']}/predictions/{phase}")
    png_path = save_path / "pngs"
    npy_path = save_path / "npys"
    png_path.mkdir(parents=True, exist_ok=True)
    npy_path.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(predictions):
        x = batch["x"].cpu().numpy()[0, :, :, 128, 0]
        y = batch["y"][0]
        yhat = batch["yhat"][0]

        np.savez(
            npy_path / f"{str(hparams[f'{phase}_ids'][i]).split('.')[0]}_prediction",
            x=batch["x"].cpu().numpy().squeeze(),
            y=y.cpu().numpy().squeeze(),
            yhat=yhat.cpu().numpy().squeeze())
        #np.save(npy_path / f"{str(hparams['test_ids'][i]).split['.'][0]}_t{j}_prediction", yhat)
        fig, ax = plt.subplots(1, 3)
        y_plot = ax[1].imshow(y.cpu().numpy()[0, :, :, 128], cmap="inferno")
        # ax[1].imshow(x, cmap="Grays_r")
        ax[1].set_title("Ground Truth")
        cmin, cmax = y_plot.get_clim()
        ax[1].axis(False)

        yhat_plot = ax[2].imshow(
            yhat.cpu().numpy()[:, :, 128, 0],
            vmin=cmin, vmax=cmax, cmap="inferno",
        )
        # ax[2].imshow(x, cmap="Grays_r")
        ax[2].set_title("Prediction")
        ax[2].axis(False)
        ax[0].imshow(x, cmap="Grays_r")
        fig.colorbar(yhat_plot, fraction=0.046, pad=0.04)

        fig.suptitle(f"Sample {hparams[f'{phase}_ids'][i]:04}")
        fig.savefig(
            png_path / f"{str(hparams[f'{phase}_ids'][i]).split('.')[0]}_prediction.png")
        plt.close()
