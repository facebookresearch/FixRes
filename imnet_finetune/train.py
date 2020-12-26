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
import torch.optim as optim
import attr
from torchvision import datasets
import torchvision.models as models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import numpy as np
from .config import TrainerConfig, ClusterConfig
from .transforms import get_transforms
from .resnext_wsl import resnext101_32x48d_wsl
from .pnasnet import pnasnet5large
try:
    from timm.models import create_model #From: https://github.com/rwightman/pytorch-image-models
    from timm.models.efficientnet import default_cfgs
    has_timm = True
except ImportError:
    has_timm = False

    

@attr.s(auto_attribs=True)
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    epoch: int
    accuracy:float
    model: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["lr_scheduler"] = self.lr_scheduler.state_dict()
        data["accuracy"] = self.accuracy
        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "TrainerState") -> "TrainerState":
        data = torch.load(filename)
        # We need this default to load the state dict
        model = default.model
        model.load_state_dict(data["model"])
        data["model"] = model

        optimizer = default.optimizer
        optimizer.load_state_dict(data["optimizer"])
        data["optimizer"] = optimizer

        lr_scheduler = default.lr_scheduler
        lr_scheduler.load_state_dict(data["lr_scheduler"])
        data["lr_scheduler"] = lr_scheduler
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
        
        Input_size_Image=self._train_cfg.input_size
        
        print("Input size : "+str(Input_size_Image))
        print("Model : " + str(self._train_cfg.architecture) )
        backbone_architecture=None
        if  self._train_cfg.architecture=='PNASNet' :
            backbone_architecture='pnasnet5large'
            
        transformation=get_transforms(input_size=self._train_cfg.input_size,test_size=self._train_cfg.input_size, kind='full', crop=True, need=('train', 'val'), backbone=backbone_architecture)
        transform_test = transformation['val']
        
        
        train_set = datasets.ImageFolder(self._train_cfg.imnet_path+ '/train',transform=transform_test)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,num_replicas=self._train_cfg.num_tasks, rank=self._train_cfg.global_rank
        )
        
        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self._train_cfg.batch_per_gpu,
            num_workers=(self._train_cfg.workers-1),
            sampler=train_sampler,
        )
        test_set = datasets.ImageFolder(self._train_cfg.imnet_path  + '/val',transform=transform_test)


        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False, num_workers=(self._train_cfg.workers-1),
        )

        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks}", flush=True)

        print("Create distributed model", flush=True)
        
        if self._train_cfg.architecture=='PNASNet' :
            model= pnasnet5large(pretrained='imagenet')
            
        if self._train_cfg.architecture=='ResNet50' :
            model=models.resnet50(pretrained=False)
            pretrained_dict=torch.load(self._train_cfg.resnet_weight_path,map_location='cpu')['model']
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
            
        if self._train_cfg.architecture=='IGAM_Resnext101_32x48d' :
            model=resnext101_32x48d_wsl(progress=True)

        if self._train_cfg.architecture=='PNASNet' :
            for name, child in model.named_children():
                if 'last_linear' not in name and 'cell_11' not in name and 'cell_10' not in name and 'cell_9' not in name:
                    for name2, params in child.named_parameters():
                        params.requires_grad = False
        elif not self._train_cfg.architecture=='EfficientNet' :
            
            for name, child in model.named_children():
                if 'fc' not in name:
                    for name2, params in child.named_parameters():
                        params.requires_grad = False
    
        if self._train_cfg.architecture=='EfficientNet' :
            assert has_timm
            model = create_model(self._train_cfg.EfficientNet_models,pretrained=False,num_classes=1000) #see https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py for name
            for name, child in model.named_children():
                if 'classifier' not in name:
                    for name2, params in child.named_parameters():
                        params.requires_grad = False
                        
            pretrained_dict=load_state_dict_from_url(default_cfgs[self._train_cfg.EfficientNet_models]['url'],map_location='cpu')
            model_dict = model.state_dict()
            for k in model_dict.keys():
                if(k in pretrained_dict.keys()):
                    model_dict[k]=pretrained_dict.get(k)
            model.load_state_dict(model_dict)
            torch.cuda.empty_cache()
            model.classifier.requires_grad=True
            model.conv_head.requires_grad=True
            
        model.cuda(self._train_cfg.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self._train_cfg.local_rank], output_device=self._train_cfg.local_rank
        )
        linear_scaled_lr = 8.0 * self._train_cfg.lr * self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks /512.0
        optimizer = optim.SGD(model.parameters(), lr=linear_scaled_lr, momentum=0.9,weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
        self._state = TrainerState(
            epoch=0,accuracy=0.0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )
        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), "checkpoint.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)

    def _train(self) -> Optional[float]:
        criterion = nn.CrossEntropyLoss()
        print_freq = 10
        acc = None
        max_accuracy=0.0
        
        print("Evaluation before fine-tuning")        
        correct = 0
        total = 0
        count=0.0
        running_val_loss = 0.0
        self._state.model.eval()
        
        if self._train_cfg.architecture=='PNASNet' :
            self._state.model.module.cell_11.eval()
            self._state.model.module.cell_10.eval()
            self._state.model.module.cell_9.eval()
            self._state.model.module.dropout.eval()
        elif self._train_cfg.architecture=='EfficientNet' :
            self._state.model.module.classifier.eval()
            self._state.model.module.conv_head.eval()
            self._state.model.module.bn2.eval()
            
        else:
            self._state.model.module.layer4[2].bn3.eval()
            
            
        with torch.no_grad():
            for data in self._test_loader:
                images, labels = data
                images = images.cuda(self._train_cfg.local_rank, non_blocking=True)
                labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)
                outputs = self._state.model(images)
                loss_val = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_val_loss += loss_val.item()
                count=count+1.0

        acc = correct / total
        ls_nm=running_val_loss/count
        print(f"Accuracy of the network on the 50000 test images: {acc:.1%}", flush=True)
        print(f"Loss of the network on the 50000 test images: {ls_nm:.3f}", flush=True)
        print("Accuracy before fine-tuning : "+str(acc))
        max_accuracy=np.max((max_accuracy,acc))
        start_epoch = self._state.epoch
        # Start from the loaded epoch
        for epoch in range(start_epoch, self._train_cfg.epochs):
            print(f"Start epoch {epoch}", flush=True)
            self._state.model.eval()
            if self._train_cfg.architecture=='PNASNet' :
                self._state.model.module.cell_11.train()
                self._state.model.module.cell_10.train()
                self._state.model.module.cell_9.train()
                self._state.model.module.dropout.train()
            elif self._train_cfg.architecture=='EfficientNet' :
                self._state.model.module.classifier.train()
                self._state.model.module.conv_head.train()
                self._state.model.module.bn2.train()
            else:
                self._state.model.module.layer4[2].bn3.train()
                
                
            self._state.lr_scheduler.step(epoch)
            self._state.epoch = epoch
            running_loss = 0.0
            for i, data in enumerate(self._train_loader):
                inputs, labels = data
                inputs = inputs.cuda(self._train_cfg.local_rank, non_blocking=True)
                labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)

                outputs = self._state.model(inputs)
                loss = criterion(outputs, labels)

                self._state.optimizer.zero_grad()
                loss.backward()
                self._state.optimizer.step()

                running_loss += loss.item()
                if i % print_freq == print_freq - 1:
                    print(f"[{epoch:02d}, {i:05d}] loss: {running_loss/print_freq:.3f}", flush=True)
                    running_loss = 0.0
                
                
            if epoch==self._train_cfg.epochs-1:
                print("Start evaluation of the model", flush=True)
                
                correct = 0
                total = 0
                count=0.0
                running_val_loss = 0.0
                self._state.model.eval()
                
                if self._train_cfg.architecture=='PNASNet' :
                    self._state.model.module.cell_11.eval()
                    self._state.model.module.cell_10.eval()
                    self._state.model.module.cell_9.eval()
                    self._state.model.module.dropout.eval()
                elif self._train_cfg.architecture=='EfficientNet' :
                    self._state.model.module.classifier.eval()
                    self._state.model.module.conv_head.eval()
                    self._state.model.module.bn2.eval()
                else:
                    self._state.model.module.layer4[2].bn3.eval()
                    
                with torch.no_grad():
                    for data in self._test_loader:
                        images, labels = data
                        images = images.cuda(self._train_cfg.local_rank, non_blocking=True)
                        labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)
                        outputs = self._state.model(images)
                        loss_val = criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        running_val_loss += loss_val.item()
                        count=count+1.0

                acc = correct / total
                ls_nm=running_val_loss/count
                print(f"Accuracy of the network on the 50000 test images: {acc:.1%}", flush=True)
                print(f"Loss of the network on the 50000 test images: {ls_nm:.3f}", flush=True)
                self._state.accuracy = acc
                if self._train_cfg.global_rank == 0:
                    self.checkpoint(rm_init=False)
                if epoch==self._train_cfg.epochs-1:
                    return acc



