from preload import PreloadNPZDataset, preload_partition_to_ram
import accelerate
import numpy as np

def load_latent_to_gt_actions(ds_path):
    dataset = PreloadNPZDataset(ds_path,
                                is_train=False, 
                                include_action=True, 
                                train_test_split_ratio=.65,
                                action_label='latents')
    corresponding_acts = PreloadNPZDataset(ds_path,
                                is_train=False, 
                                include_action=True, 
                                train_test_split_ratio=.65,
                                action_label='action',
                                remove_last_action=True)

    accelerator = accelerate.Accelerator()
    indices = np.arange(len(dataset))
    _, _, latent_actions = preload_partition_to_ram(dataset, accelerator, indices,fixed_size_trajectory=False, include_action=True)
    _,_, true_actions = preload_partition_to_ram(corresponding_acts, accelerator, indices,fixed_size_trajectory=False, include_action=True)

    return latent_actions, true_actions