from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data.distributed import DistributedSampler
from . import OCL_dataset
from torch.utils.data import DataLoader
import numpy as np

def list_collation(list_of_sample):
        batch = defaultdict(list)
        for sample in list_of_sample:
            for k, v in sample.items():
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                batch[k].append(v)
        return dict(batch)

def concat_collation(list_of_sample):
    batch = list_collation(list_of_sample)
    for k, v in batch.items():
        if k == 'gt_causal':
            batch[k] = []
            for i, cur_v in enumerate(v):
                if len(cur_v) == 0:
                    continue
                batch[k].append(torch.cat(
                    [torch.ones(len(cur_v), 1, dtype=int) * i, cur_v], 1
                ))
            if len(batch[k]) == 0:
                batch[k] = torch.tensor([])
            else:
                batch[k] = torch.cat(batch[k], 0)
            # N * 3
            # [ [inst_id, attr_id, aff_id] , ... ,...]
        else:
            batch[k] = torch.stack(v, 0)
    return batch




def get_dataloader(phase: str,
                   data_type: str, feature_dir: Optional[str] = None,
                   batchsize: Optional[int] = 1,
                   num_workers: Optional[int] = 0,
                   shuffle: Optional[bool] = None,
                   distributed_sampler: Optional[bool] = False,
                   ) -> DataLoader:

    if data_type == "image":
        dataset = OCL_dataset.ImageDataset(
                phase=phase)

    elif data_type == "feature":
        dataset = OCL_dataset.FeatureDataset(
                phase=phase, feature_dir=feature_dir)


    if shuffle is None:
        shuffle = (phase == 'train')

    if data_type == "image":
        collation_function = list_collation
    else:
        collation_function = concat_collation

    if distributed_sampler:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batchsize,
                          num_workers=num_workers, collate_fn=collation_function,
                          sampler=sampler)

    else:
        return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=collation_function)
