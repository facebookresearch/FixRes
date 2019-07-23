# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def run(path,name,version):
    if version=='v1':
        softmax=np.load(path+name+'_softmax.npy')
    else:
        softmax=np.load(path+name+'_softmax_v2.npy')
        
    label=np.load(path+'labels.npy')
    assert softmax.shape[0] == 50000,"Error ImageNet validation set doesn't match"
    correct=0.0
    for i in range(softmax.shape[0]):
        prediction=np.argmax(softmax[i])
        if prediction==label[i]:
            correct=correct+1.0
    acc=(correct/50000.0)*100
    print("Top-1 Accuracy: %.1f"%(acc))
    return acc
    
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script for softmax extracted from FixRes models",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--architecture', default='IGAM_Resnext101_32x48d', type=str,choices=['ResNet50' , 'ResNet50CutMix', 'PNASNet' , 'IGAM_Resnext101_32x48d'], help='Neural network architecture')
    parser.add_argument('--save-path', default='/where/are/save/softmax/', type=str, help='Path where softmax were saved')
    parser.add_argument('--version', default='v1', type=str,choices=['v1' , 'v2'], help='version')
    args = parser.parse_args()
    run(args.save_path,args.architecture,args.version)
