import torch
import os
import numpy as np
import glob
from torch.utils.data import DataLoader, Dataset, Subset
import random
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
import cv2

class PreloadedDataset(Dataset):
    def __init__(self, gpu_samples, 
                 transform=None,
                 chunk_length=2,
                 min_frame_interval=15,
                 max_frame_interval=20,
                 target_frame_boundary=40,
                 randomize_samples=False,
                 init_frame_zero=False,
                 traj_length=None,
                 augment_chunk=False,
                 fixed_size_trajectory=True,
                 chunk_indices=[],
                 last_frame_index_as_target=True,
                 ):
        self.samples = gpu_samples
        self.transform = transform
        self.chunk_length = chunk_length
        self.min_frame_interval = min_frame_interval
        self.max_frame_interval = max_frame_interval
        self.target_frame_boundary = target_frame_boundary
        self.randomize_samples = randomize_samples
        self.traj_length = traj_length
        self.init_frame_zero = init_frame_zero
        self.augment_chunk = augment_chunk
        self.fixed_size_trajectory = fixed_size_trajectory
        self.chunk_indices = chunk_indices
        self.last_frame_index_as_target = last_frame_index_as_target
        assert min_frame_interval <= max_frame_interval, 'min_frame_interval should be <= max_frame_interval'
        if not fixed_size_trajectory:
            assert len(chunk_indices) > 0, "chunk_indices must be provided for variable size trajectories"

    def __len__(self):
        return self.samples.size(0) if self.fixed_size_trajectory else self.chunk_indices.size(0) - 1

    def __getitem__(self, idx):
        sample = self.samples[idx] if self.fixed_size_trajectory else self.samples[0, self.chunk_indices[idx]:self.chunk_indices[idx + 1],...]
        num_frames = sample.shape[0]

        max_start = num_frames - (self.chunk_length + 2) * self.max_frame_interval - self.target_frame_boundary - 1
        if max_start < 0:
            raise ValueError(f"Not enough frames in the sample. Sample length: {num_frames}, "
                             f"max_start: {max_start}, chunk_length: {self.chunk_length}, "
                             f"min_frame_interval: {self.min_frame_interval}, "
                             f"max_frame_interval: {self.max_frame_interval}, "
                             f"target_frame_boundary: {self.target_frame_boundary}")
        start_idx = 0 if self.init_frame_zero else random.randint(0, max_start)

        if self.randomize_samples:
            sampled_indices = [start_idx]
            for _ in range(self.chunk_length):
                sampled_indices.append(sampled_indices[-1] + random.randint(self.min_frame_interval, self.max_frame_interval))
        else:
            interval = random.randint(self.min_frame_interval, self.max_frame_interval)
            sampled_indices = list(range(start_idx, start_idx + interval * (self.chunk_length + 1), interval))

        sampled_indices = sampled_indices[1:]
        last_frame_index = sampled_indices[-1] if sampled_indices else start_idx
        target_frame_index = last_frame_index + random.randint(self.min_frame_interval, self.target_frame_boundary)
        query_index = start_idx

        assert torch.all(query_index < torch.tensor(sampled_indices)), f"Query index {query_index} should be less than sampled indices {sampled_indices}"
        assert torch.all(target_frame_index > torch.tensor(sampled_indices)), f"Target frame index {target_frame_index} should be greater than sampled indices {sampled_indices}"
        
        frames_to_sample = torch.tensor([query_index] + sampled_indices + [target_frame_index]).to(sample.device)   
        frames = torch.index_select(sample, 0, frames_to_sample.to(sample.device))
       
        frames = self.transform(frames) if self.transform else frames

        query_frame = frames[0]
        trajectory = frames #[1:-1]
        if self.last_frame_index_as_target:
            target_frame_index = target_frame_index - query_index
            target_frame = frames[-1] 
        else:
            target_frame_index = random.randint(1, trajectory.size(0) - 1)
            target_frame = frames[target_frame_index]

        if self.augment_chunk:
            if random.random() < 0.25:
                augmentations = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
                                                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])
                trajectory = augmentations(trajectory)     

        return trajectory, query_frame, target_frame, frames_to_sample - query_index, torch.tensor(target_frame_index).to(sample.device)


def preload_partition_to_x(dataset, accelerator, indices, resize=64, fixed_size_trajectory=True, include_action=False, to_ram=False, hide_inventory=False):
    """
    Preload a partition of the dataset to RAM or GPU.
    Args:
        dataset (Dataset): The dataset to preload.
        accelerator (Accelerator): The accelerator to use.
        indices (np.ndarray): The indices of the samples to preload.
        resize (int): The size to resize the images to.
        fixed_size_trajectory (bool): If True, the trajectory size is fixed.
        include_action (bool): If True, include action data in the preloading.
        to_ram (bool): If True, preload to RAM, otherwise to GPU.
    Returns:
        preloaded_videos (torch.Tensor): The preloaded videos.
        preloaded_actions (torch.Tensor): The preloaded actions (if include_action is True).
    """
    preload_batch_size = 16 if fixed_size_trajectory else 1
    loader = DataLoader(Subset(dataset, indices), batch_size=preload_batch_size, shuffle=False, num_workers=0)
    preloaded_videos, preloaded_actions = [], []
    traj_lengths = [0]

    to = 'RAM' if to_ram else 'GPU'
    desc = f"Preloading to {to} for rank {accelerator.process_index}"

    for batch in tqdm(loader, disable=not accelerator.is_main_process, desc=desc):
        if include_action:
            batch, batch_action = batch
            preloaded_actions.append(batch_action.squeeze())#rearrange(batch_action, 'b t -> (b t)'))
            
        if hide_inventory:
            batch[..., 49:,:] = 0

        if batch.dtype == torch.uint8:
            batch = batch.to(torch.float16) / 255.0

        if batch.size(-1) == 3:
            batch = batch.permute(0, 1, 4, 2, 3)  # Convert to CHW format

        if batch.size(-1) != resize:
            b = batch.size(0)
            batch = rearrange(batch, 'b t c h w -> (b t) c h w')
            batch = F.interpolate(batch, size=(resize, resize), mode='bilinear', align_corners=False)
            batch = rearrange(batch, '(b t) c h w -> b t c h w', b=b)

        batch_on_device = batch.to(accelerator.device, non_blocking=True) if not to_ram else batch
        preloaded_videos.append(batch_on_device)
        
        if not fixed_size_trajectory:       
            traj_lengths.append(batch_on_device.shape[1] + traj_lengths[-1])
    if not fixed_size_trajectory: 
        traj_lengths = torch.tensor(traj_lengths, device=accelerator.device) if not to_ram else torch.tensor(traj_lengths)
        preloaded_videos = torch.cat(preloaded_videos, dim=1)
    else:
        preloaded_videos = torch.cat(preloaded_videos, dim=0)
        traj_lengths = None
    action_lbls = torch.cat(preloaded_actions, dim=0) if include_action else None    

    return preloaded_videos, traj_lengths, action_lbls


class PreloadNPZDataset(Dataset):
    def __init__(self, path, 
                    separate_train_test_flds=False, 
                    is_train=True, 
                    train_test_split_ratio=0.8, 
                    include_action=False, 
                    appended_path='',
                    action_label='action',
                    remove_last_action=False,
                    policy_training=False):     

        self.path = path if not separate_train_test_flds else os.path.join(path, 'train' if is_train else 'test')
        self.path = os.path.join(self.path, appended_path)
        self.include_action = include_action
        self.action_label = action_label
        self.npz_files = glob.glob(os.path.join(self.path, '*.npz'))
        self.npz_files.sort()
        self.remove_last_action = remove_last_action
        split_idx = len(self.npz_files) if separate_train_test_flds else int(len(self.npz_files) * train_test_split_ratio)
        if not separate_train_test_flds:
            self.npz_files = self.npz_files[:split_idx] if is_train else self.npz_files[split_idx:]
        self.policy_training = policy_training
    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        data_array = torch.tensor(data['video'])
        if self.include_action:
            if 'latent' in self.action_label:
                data_array_action = data['latents']
                if self.policy_training:
                    terminal_act = np.zeros((1, data_array_action.shape[1], data_array_action.shape[2]), dtype=data_array_action.dtype)
                    data_array_action = np.concatenate([data_array_action, terminal_act], axis=0)
                # assert data_array.shape[0] - 1 == data_array_action.shape[0], f'{npz_file}'
            else:
                data_array_action = data[self.action_label][1:] if self.remove_last_action else data[self.action_label]
                assert data_array.shape[0] - 1 == data_array_action.shape[0], f'{npz_file}'
            data_array_action = torch.tensor(data_array_action)
            return data_array, data_array_action
        return data_array

class PreloadMP4Dataset(Dataset):
    def __init__(self, path, appended_path='train'):
        self.path = path
        self.video_files = glob.glob(os.path.join(path, appended_path, '*.mp4'))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            frames.append(frame)
        cap.release()
        print(f"Loaded {len(frames)} frames from {video_path}")
        video_array = np.stack(frames)  # shape: (num_frames, H, W, 3)
        return torch.tensor(video_array, dtype=torch.uint8)


def preload_partition_to_gpu(dataset, accelerator, indices, resize=64, fixed_size_trajectory=True, include_action=False, hide_inventory=False):
    return preload_partition_to_x(dataset, accelerator, indices, resize=resize, fixed_size_trajectory=fixed_size_trajectory, include_action=include_action, to_ram=False, hide_inventory=hide_inventory)

def preload_partition_to_ram(dataset, accelerator, indices, resize=64, fixed_size_trajectory=True, include_action=False, hide_inventory=False):
    return preload_partition_to_x(dataset, accelerator, indices, resize=resize, fixed_size_trajectory=fixed_size_trajectory, include_action=include_action, to_ram=True, hide_inventory=hide_inventory)