import torch
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced, TrajSlicerDataset
from typing import Optional, Callable, Any
import glob
decord.bridge.set_bridge("torch")

class CrafterDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/crafter",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = False
        self.proprio_dim = 0
        self.state_dim = 0
        
        print("Loading crafter dataset...")

        self.all_files = glob.glob(str(self.data_path/ "*.npz"))
        self.all_files.sort()
        action_data = []
        latent_action_data = []
        self.seq_lengths = []
        for file in self.all_files:
            data = np.load(file)
            action_data.append(data['action'])
            latent_action_data.append(data['latents'])
            self.seq_lengths.append(action_data[-1].shape[0])
        
        actions = torch.from_numpy(np.concatenate(action_data)).long()  # discrete actions therefore no need to scale
        latent_actions = torch.from_numpy(np.concatenate(latent_action_data)).float()
        print(f"Loaded {len(self.all_files)} rollouts")

        self.action_dim = actions.shape[-1]
        self.latent_action_dim = latent_actions.shape[-1]

    def get_all_actions(self):
        return self.actions

    def get_frames(self, idx, frames):
        file = self.all_files[idx]
        data = np.load(file)
        action = torch.from_numpy(data['action']).long()
        latent_action = data['latents']
        image = data['video'][frames]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        image = rearrange(image, "T H W C -> T C H W")
        torch_image = torch.from_numpy(image)
        torch_latent_action = torch.from_numpy(latent_action)
        if self.transform:
            torch_image = self.transform(torch_image)
        obs = {
            "visual": torch_image
        }
        return obs, (torch_latent_action, action), {}, {}
    
    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)



def load_crafter_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/crafter',
    normalize_action=False,
    split_ratio=0.8,
    split_mode="random",
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    if split_mode == "random":
        dset = CrafterDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
        )
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset, 
            train_fraction=split_ratio, 
            num_frames=num_hist + num_pred, 
            frameskip=frameskip
        )
    elif split_mode == "folder":
        dset_train = CrafterDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/train",
            normalize_action=normalize_action,
        )
        dset_val = CrafterDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/val",
            normalize_action=normalize_action,
        )
        num_frames = num_hist + num_pred
        train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
        val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset

