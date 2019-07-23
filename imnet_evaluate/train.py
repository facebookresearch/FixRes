# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import os.path as osp
from typing import Optional
import torch
import torch.distributed
import torch.nn as nn
import attr
from torchvision import datasets
import tqdm
import torchvision.models as models
import numpy as np
from .config import TrainerConfig, ClusterConfig
from .transforms import get_transforms
from .resnext_wsl import resnext101_32x48d_wsl
from collections import defaultdict
from .pnasnet import pnasnet5large


def accuracy_sp(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k * (100.0 / batch_size))
    return res

class HistoryMeter(object):
    """Remember all values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []
        self.partials = []
        self.count = 0
        self.val = 0

    def update(self, x, n=1):
        self.val = x
        self.hist.append(x)
        x = n * x
        self.count += n
        # full precision summation based on http://code.activestate.com/recipes/393090/
        i = 0
        for y in self.partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                self.partials[i] = lo
                i += 1
            x = hi
        self.partials[i:] = [x]

    @property
    def avg(self):
        """
        Alternative to AverageMeter without floating point errors
        """
        return sum(self.partials, 0.0) / self.count if self.partials else 0




@attr.s(auto_attribs=True)
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    model: nn.Module

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data["model"] = self.model.state_dict()

        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "TrainerState") -> "TrainerState":
        data = torch.load(filename)
        # We need this default to load the state dict
        model = default.model
        model.load_state_dict(data["model"])
        data["model"] = model

        return cls(**data)


class Trainer:
    def __init__(self, train_cfg: TrainerConfig, cluster_cfg: ClusterConfig) -> None:
        self._train_cfg = train_cfg
        self._cluster_cfg = cluster_cfg

    def __call__(self) -> Optional[float]:
        """
        Called for each task.

        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        self._init_state()
        final_acc = self._train()
        return final_acc

    def checkpoint(self, rm_init=True):
        
        save_dir = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id))
        os.makedirs(save_dir, exist_ok=True)
        self._state.save(osp.join(save_dir, "checkpoint.pth"))
        self._state.save(osp.join(save_dir, "checkpoint_"+str(self._state.epoch)+".pth"))
        if rm_init:
            os.remove(self._cluster_cfg.dist_url[7:])
        empty_trainer = Trainer(self._train_cfg, self._cluster_cfg)
        return empty_trainer

    def _setup_process_group(self) -> None:
        
        torch.cuda.set_device(self._train_cfg.local_rank)
        torch.distributed.init_process_group(
            backend=self._cluster_cfg.dist_backend,
            init_method=self._cluster_cfg.dist_url,
            world_size=self._train_cfg.num_tasks,
            rank=self._train_cfg.global_rank,
        )
        print(f"Process group: {self._train_cfg.num_tasks} tasks, rank: {self._train_cfg.global_rank}")

    def _init_state(self) -> None:
        """
        Initialize the state and load it from an existing checkpoint if any
        """
        torch.manual_seed(0)
        np.random.seed(0)
        
        print("Create data loaders", flush=True)
        print("Input size : "+str(self._train_cfg.input_size))
        print("Model : " + str(self._train_cfg.architecture) )
        backbone_architecture=None
        if  self._train_cfg.architecture=='PNASNet' :
            backbone_architecture='pnasnet5large'
            
        
        transformation=get_transforms(input_size=self._train_cfg.input_size,test_size=self._train_cfg.input_size, kind='full', crop=True, need=('train', 'val'), backbone=backbone_architecture)
        transform_test = transformation['val']

        test_set = datasets.ImageFolder(self._train_cfg.imnet_path + '/val',transform=transform_test)

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False, num_workers=(self._train_cfg.workers-1),
        )


        print("Create distributed model", flush=True)
        
        if self._train_cfg.architecture=='PNASNet' :
            model= pnasnet5large(pretrained='imagenet')
            
        if self._train_cfg.architecture=='ResNet50' :
            model=models.resnet50(pretrained=False)
            
        if self._train_cfg.architecture=='IGAM_Resnext101_32x48d' :
            model=resnext101_32x48d_wsl(progress=True)

        pretrained_dict=torch.load(self._train_cfg.weight_path,map_location='cpu')['model']
        model_dict = model.state_dict()
        count=0
        count2=0
        for k in model_dict.keys():
            count=count+1.0
            if(('module.'+k) in pretrained_dict.keys()):
                count2=count2+1.0
                model_dict[k]=pretrained_dict.get(('module.'+k))
        model.load_state_dict(model_dict)
        print("load "+str(count2*100/count)+" %")
        
        assert int(count2*100/count)== 100,"model loading error"
        
        for name, child in model.named_children():
            for name2, params in child.named_parameters():
                params.requires_grad = False
    
        print('model_load')
        if torch.cuda.is_available():
            model.cuda(self._train_cfg.local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self._train_cfg.local_rank], output_device=self._train_cfg.local_rank
            )
        
        self._state = TrainerState(
             model=model
        )
        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), "checkpoint.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)

    def _train(self) -> Optional[float]:
        
        self._state.model.eval()
        metrics = defaultdict(HistoryMeter)
        with torch.no_grad():
            for data in tqdm.tqdm(self._test_loader):
                images, labels = data
                images = images.cuda(self._train_cfg.local_rank, non_blocking=True)
                labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)
                outputs = self._state.model(images)
                top1, top5 = accuracy_sp(outputs, labels, topk=(1, 5))
                metrics["val_top1"].update(top1, n=images.size(0))
                metrics["val_top5"].update(top5, n=images.size(0))
        for k in metrics: metrics[k] = metrics[k].avg
        print(metrics)
        return metrics['val_top1'],metrics['val_top5']



