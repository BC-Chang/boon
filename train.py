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

torch.set_float32_matmul_precision('medium')

seed = 1758938
torch.manual_seed(0)
np.random.seed(0)

# Load data
datapath = Path("/scratch/06898/bchang/elec_sims")  # Path(r"C:\Users\bchan\Box\1001_datasets\elec_sims")
samples = np.array(list(datapath.glob("494*")))
n_samples = 100  # len(samples)
split = [0.6, 0.2, 0.2]
assert len(split) == 3, "Split must be a list of three floats"
assert sum(split) == 1, "Split must sum to 1"

image_ids = np.random.choice(np.arange(len(samples)), size=n_samples, replace=False)
train_ids, val_ids, test_ids = split_indices(np.arange(n_samples), split, seed=seed)

train_ids = np.array([samples[image_ids[ids]].name for ids in train_ids])
val_ids = np.array([samples[image_ids[ids]].name for ids in val_ids])
test_ids = np.array([samples[image_ids[ids]].name for ids in test_ids])


hparams = {
    'net_name': "BOON_v0.0.10",
    'learning_rate': 1e-3,
    'batch_size': 5,
    'epochs': 300,
    'val_interval': 10,
    'modes1': 8,
    'modes2': 8,
    'modes3': 8,
    'width': 20,
    'num_layers': 3,
    'beta_1': 1,
    'beta_2': 0,
    'n_samples': n_samples,
    'n_train': len(train_ids),
    'n_val': len(val_ids),
    'n_test': len(test_ids),
    'train_ids': [str(train_id) for train_id in train_ids],
    'val_ids': [str(val_id) for val_id in val_ids],
    'test_ids': [str(test_id) for test_id in test_ids],
    'patience': 30
}
wandb.init(project="BOON-FNO", name=hparams['net_name'],
            config=hparams, save_code=True, id=hparams['net_name'])

logger = WandbLogger()
run_id = logger.experiment.id


train_dataloader = get_dataloader(
    train_ids,
    image_size=(256, 256, 256),
    data_path=datapath,
    phase='train',
    num_workers=71,
#    persistent_workers=False,
)

val_dataloader = get_dataloader(
    val_ids,
    image_size=(256, 256, 256),
    data_path=datapath,
    phase='val',
    num_workers=71,
#    persistent_workers=False,
)

try:
    model_dir = f"BOON-FNO/{hparams['net_name']}/checkpoints"
    model_loc = glob.glob(f"{model_dir}/*val*.ckpt")[0]
    print(f"Loading {model_loc}")
    model = BOON3D.load_from_checkpoint(model_loc)

except IndexError:
    # Instantiate the model
    print("Instantiating a new model...")
    model = BOON3D(
        net_name=hparams['net_name'],
        in_channels=1,
        out_channels=1,
        modes1=hparams['modes1'],
        modes2=hparams['modes2'],
        modes3=hparams['modes3'],
        width=hparams['width'],
        num_layers=hparams['num_layers'],
        lr=hparams['learning_rate'],
    )
    log_path = Path(f"./lightning_logs/{hparams['net_name']}")
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "hparam_config.json", 'w') as f:
        json.dump(hparams, f)

# Add some checkpointing callbacks
cbs = [
    # ModelCheckpoint(
    #     monitor="loss",
    #     filename="{epoch:02d}-{loss:.2f}",
    #     dirpath=logger.log_dir,
    #     save_top_k=1,
    #     mode="min",
    # ),
    ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
        dirpath=logger.log_dir,
        save_top_k=1,
        mode="min",
    ),
    LearningRateMonitor(logging_interval='epoch',
                        log_momentum=True, log_weight_decay=True),
    GradientAccumulationScheduler(scheduling={0: hparams['batch_size']}),
    EarlyStopping(monitor="val_loss", check_finite=False,
                    patience=hparams['patience']),
]

trainer = pl.Trainer(
    callbacks=cbs,  # Add the checkpoint callback
    max_epochs=hparams['epochs'],
    check_val_every_n_epoch=hparams['val_interval'],
    log_every_n_steps=hparams['n_train'],
    logger=logger,
    precision="bf16-mixed",
)

trainer.fit(model, train_dataloader, val_dataloader)

wandb.finish()

