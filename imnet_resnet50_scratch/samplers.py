# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.utils.data.sampler import BatchSampler
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
import math
import torch.distributed as dist

class RASampler(torch.utils.data.Sampler):
    """
    Batch Sampler with Repeated Augmentations (RA)
    - dataset_len: original length of the dataset
    - batch_size
    - repetitions: instances per image
    - len_factor: multiplicative factor for epoch size
    """

    def __init__(self,dataset,num_replicas, rank, dataset_len, batch_size, repetitions=1, len_factor=1.0, shuffle=False, drop_last=False):
        self.dataset=dataset
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.repetitions = repetitions
        self.len_images = int(dataset_len * len_factor)
        self.shuffle = shuffle
        self.drop_last = drop_last
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.repetitions * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        
        
    def shuffler(self):
        if self.shuffle:
            new_perm = lambda: iter(np.random.permutation(self.dataset_len))
        else:
            new_perm = lambda: iter(np.arange(self.dataset_len))
        shuffle = new_perm()
        while True:
            try:
                index = next(shuffle)
            except StopIteration:
                shuffle = new_perm()
                index = next(shuffle)
            for repetition in range(self.repetitions):
                yield index

    def __iter__(self):
        shuffle = iter(self.shuffler())
        seen = 0
        indices=[]
        for _ in range(self.len_images):
            index = next(shuffle)
            indices.append(index)
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        
def list_collate(batch):
    """
    Collate into a list instead of a tensor to deal with variable-sized inputs
    """
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy':
        if elem_type.__name__ == 'ndarray':
            return list_collate([torch.from_numpy(b) for b in batch])
    elif isinstance(batch[0], Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    return default_collate(batch)
