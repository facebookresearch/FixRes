# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

import numpy as np

class Resize(transforms.Resize):
    """
    Resize with a ``largest=False'' argument
    allowing to resize to a common largest side without cropping
    """


    def __init__(self, size, largest=False, **kwargs):
        super().__init__(size, **kwargs)
        self.largest = largest

    @staticmethod
    def target_size(w, h, size, largest=False):
        if h < w and largest:
            w, h = size, int(size * h / w)
        else:
            w, h = int(size * w / h), size
        size = (h, w)
        return size

    def __call__(self, img):
        size = self.size
        w, h = img.size
        target_size = self.target_size(w, h, size, self.largest)
        return F.resize(img, target_size, self.interpolation)

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ', largest={})'.format(self.largest)




def get_transforms(input_size=224,test_size=224, kind='full', crop=True, need=('train', 'val'), backbone=None):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    transformations = {}
    if 'train' in need:
        if kind == 'torch':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif kind == 'full':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.ToTensor(),            
                transforms.Normalize(mean, std),
            ])

        else:
            raise ValueError('Transforms kind {} unknown'.format(kind))
    if 'val' in need:
        if crop:
            transformations['val'] = transforms.Compose(
                [Resize(int((256 / 224) * test_size)),  # to maintain same ratio w.r.t. 224 images
                 transforms.CenterCrop(test_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            transformations['val'] = transforms.Compose(
                [Resize(test_size, largest=True),  # to maintain same ratio w.r.t. 224 images
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
    return transformations

transforms_list = ['torch', 'full']
