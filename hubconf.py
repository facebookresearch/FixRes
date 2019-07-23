# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from tqdm import tqdm
import torch
import hashlib
import os
import re
import shutil
import sys
import tempfile

try:
    from requests.utils import urlparse
    from requests import get as urlopen
    requests_available = True
except ImportError:
    requests_available = False
    if sys.version_info[0] == 2:
        from urlparse import urlparse  # noqa f811
        from urllib2 import urlopen  # noqa f811
    else:
        from urllib.request import urlopen
        from urllib.parse import urlparse
        
dependencies = ['torch', 'torchvision']

from torchvision.models.resnet import ResNet, Bottleneck

def _download_url_to_file(url, dst, hash_prefix, progress):
    r"""
    function from https://pytorch.org/docs/stable/model_zoo.html
    """
    if requests_available:
        u = urlopen(url, stream=True)
        file_size = int(u.headers["Content-Length"])
        u = u.raw
    else:
        u = urlopen(url)
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta.get_all("Content-Length")[0])

    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
            
def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True):
    r"""
    function from https://pytorch.org/docs/stable/model_zoo.html
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None 
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return torch.load(cached_file, map_location=map_location)
    
    
model_urls = {
    'FixResNet50': 'https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNet50_v2.pth',
    'FixResNet50CutMix': 'https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNet50_CutMix_v2.pth',
    'FixResNeXt101_32x48d': 'https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNext101_32x48d_v2.pth',
}


def _fixmodel(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location='cpu')['model']
    model_dict = model.state_dict()
    count=0
    count2=0
    for k in model_dict.keys():
        count=count+1.0
        if(('module.'+k) in pretrained_dict.keys()):
            count2=count2+1.0
            model_dict[k]=pretrained_dict.get(('module.'+k))
            
    assert int(count2*100/count)== 100,"model loading error"
    
    model.load_state_dict(model_dict)
    return model

def fixresnet_50(progress=True, **kwargs):
    """Constructs a FixResNet-50 
    `"Fixing the train-test resolution discrepancy" <https://arxiv.org/abs/1906.06423>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """

    return _fixmodel('FixResNet50', Bottleneck, [3, 4, 6, 3], True, progress, **kwargs)

def fixresnet_50_CutMix(progress=True, **kwargs):
    """Constructs a FixRes-50 CutMix 
    `"Fixing the train-test resolution discrepancy" <https://arxiv.org/abs/1906.06423>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _fixmodel('FixResNet50CutMix', Bottleneck, [3, 4, 6, 3], True, progress, **kwargs)

def fixresnext101_32x48d(progress=True, **kwargs):
    """Constructs a FixResNeXt-101 32x48 
    `"Fixing the train-test resolution discrepancy" <https://arxiv.org/abs/1906.06423>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    return _fixmodel('FixResNeXt101_32x48d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)
