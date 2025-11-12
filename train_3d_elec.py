import sys
from pathlib import Path

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

from dataloader import get_dataloader, split_indices

seed = 1758938
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
datapath = Path(r"C:\Users\bchan\Box\1001_datasets\elec_sims")  # Path("/scratch/06898/bchang/elec_sims")
samples = np.array(list(datapath.glob("494*")))
n_samples = 10  # len(samples)
split = [0.6, 0.2, 0.2]
assert len(split) == 3, "Split must be a list of three floats"
assert sum(split) == 1, "Split must sum to 1"

image_ids = np.random.choice(np.arange(len(samples)), size=n_samples, replace=False)
train_ids, val_ids, test_ids = split_indices(np.arange(n_samples), split, seed=seed)

train_ids = np.array([samples[image_ids[ids]].name for ids in train_ids])
val_ids = np.array([samples[image_ids[ids]].name for ids in val_ids])
test_ids = np.array([samples[image_ids[ids]].name for ids in test_ids])

train_dataloader = get_dataloader(
    train_ids,
    image_size=(256, 256, 256),
    data_path=datapath, #"D:/deeprock_electrical/carbonate",  # "D:/sandstone_simulations/elec_sims",  #
    phase='train',
    num_workers=0,
    persistent_workers=False,
)

val_dataloader = get_dataloader(
    val_ids,
    image_size=(256, 256, 256),
    data_path=datapath, #"D:/deeprock_electrical/carbonate",  #"D:/sandstone_simulations/elec_sims",
    phase='val',
    num_workers=0,
    persistent_workers=False,
)

# Training parameters
modes = 8
width = 20

epochs = 50
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

sub = 2
N = 100 // sub #total grid size divided by the subsampling rate
S = N

base_no = FNO3d(modes, modes, modes, width)
model = BOON_FNO3d(width,
                    base_no,
                    bdy_type = 'dirichlet').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_dataloader:
        bs, nx, ny, nz, _ = x.shape
        x, y = x.to(device), y.to(device)
        print(y.shape)
        optimizer.zero_grad()
        
        bdy_left  = y[:, :, 0, :, :].reshape(bs, 1, ny, nz)
        bdy_right = y[:,:, -1, :, :].reshape(bs, 1, ny, nz)
        # bdy_top   = y[:, :, :, 0, :].reshape(bs, 1, nx, nz)
        # bdy_down  = y[:, :, :,-1, :].reshape(bs, 1, nx, nz)
        
        out = model(x, 
                    bdy_left = {'val':bdy_left}, 
                    bdy_right = {'val':bdy_right}, 
                    bdy_top = {'val':bdy_top}, 
                    bdy_down = {'val':bdy_down}
                ).view(bs, S, S, nz)

        l2 = myloss(out.view(bs, -1), y.view(bs, -1))
        l2.backward()

        optimizer.step()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in val_dataloader:
            bs, nx, ny, nz, _ = x.shape
            x, y = x.to(device), y.to(device)
            
            bdy_left  = y[:, 0, :, :].reshape(bs, 1, ny, nz) # add extra dimension to take care of 
#                                                         model channel structure
            bdy_right = y[:,-1, :, :].reshape(bs, 1, ny, nz)
            bdy_top   = y[:, :, 0, :].reshape(bs, 1, nx, nz)
            bdy_down  = y[:, :,-1, :].reshape(bs, 1, nx, nz)

            out = model(x,
                    bdy_left = {'val':bdy_left}, 
                    bdy_right = {'val':bdy_right}, 
                     bdy_top = {'val':bdy_top}, 
                    bdy_down = {'val':bdy_down}
                ).view(bs, S, S, nz)
            test_l2 += myloss(out.view(bs, -1), y.view(bs, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)
# torch.save(model, path_model)