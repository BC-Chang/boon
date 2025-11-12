import numpy as np
import torch
from hdf5storage import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

_MIN = 0
_MAX = 1000

class NumpyDataset(Dataset):
    def __init__(self, image_ids, image_size, data_dir):
        self.image_ids = image_ids
        self.image_size = image_size
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        try:
            # Load data from .npy files
            input_field = np.fromfile(self.data_dir / f"{self.image_ids[idx]:04}" / "segmented.raw", dtype=np.uint8).reshape((self.image_size[0], self.image_size[1], self.image_size[2], 1))
            input_field = input_field[:100, :100, :100, :]
            # TODO: Add linear trend to input field
            
            # Potential Fields
            phi = np.fromfile(self.data_dir / f"{self.image_ids[idx]:04}" / "elecpot.raw", dtype=np.float32).reshape((1, self.image_size[0], self.image_size[1], self.image_size[2]))
            phi = phi[:, :100, :100, :100]

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {e} not found.")
        
        input_field = torch.from_numpy(input_field).float()
        phi = torch.from_numpy(phi)

        return input_field, phi
    
    def normalize_grayscale(self, x: np.ndarray):
        """Normalize the unique gray levels to [0, 1]

        Parameters:
        ---
            x: np.ndarray of unique gray levels (for uint8: 0-255)
        """
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))

def get_dataloader(
    image_ids,
    data_path,
    phase,
    image_size,
    batch=1,
    num_workers=2,
    **kwargs,
):

    # print(f"{train_ids = }\n{val_ids = }\n{test_ids = }")

    dataset = NumpyDataset(
        image_ids=image_ids, data_dir=data_path, image_size=image_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=(phase == 'train'),
        # persistent_workers=True,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )

    return dataloader

def split_indices(indices, split, seed=None):
    if seed is not None:
        np.random.seed(seed)

    assert len(split) == 3, "Split must be a list of length 3."
    assert sum(split) == 1.0, "Sum of split must equal one."

    np.random.shuffle(indices)
    train_size = int(split[0] * len(indices))
    val_size = int(split[1] * len(indices))

    train_ids = indices[:train_size]
    val_ids = indices[train_size: (val_size + train_size)]
    test_ids = indices[(val_size + train_size):]

    return train_ids, val_ids, test_ids
