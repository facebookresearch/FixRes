# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import NamedTuple


class ClusterConfig(NamedTuple):
    dist_backend: str
    dist_url: str


class TrainerConfig(NamedTuple):
    data_folder: str
    epochs: int
    lr: float
    input_size: int
    batch_per_gpu: int
    save_folder: str
    imnet_path: str
    workers: int
    local_rank: int
    global_rank: int
    num_tasks: int
    job_id: str
