# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import uuid
from pathlib import Path
from imnet_resnet50_scratch import TrainerConfig, ClusterConfig, Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def run(input_sizes,learning_rate,epochs,batch,node,workers,imnet_path,shared_folder_path,job_id,local_rank,global_rank,num_tasks):
    cluster_cfg = ClusterConfig(dist_backend="nccl", dist_url="")
    shared_folder=None
    data_folder_Path=None
    if Path(str(shared_folder_path)).is_dir():
        shared_folder=Path(shared_folder_path+"/training/")
    else:
        raise RuntimeError("No shared folder available")
    if Path(str(imnet_path)).is_dir():
        data_folder_Path=Path(str(imnet_path))
    else:
        raise RuntimeError("No shared folder available")
    train_cfg = TrainerConfig(
                    data_folder=str(data_folder_Path),
                    epochs=epochs,
                    lr=learning_rate,
                    input_size=input_sizes,
                    batch_per_gpu=batch,
                    save_folder=str(shared_folder_path),
                    workers=workers,
                    imnet_path=imnet_path,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    num_tasks=num_tasks,
                    job_id=job_id,
                )
        
    # Create the executor
    os.makedirs(str(shared_folder), exist_ok=True)
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
        
    cluster_cfg = cluster_cfg._replace(dist_url=init_file.as_uri())
    trainer = Trainer(train_cfg, cluster_cfg)
    
    #The code should be launch on each GPUs
    try:    
        if global_rank==0:
            val_accuracy = trainer.__call__()
            print(f"Validation accuracy: {val_accuracy}")
        else:
            trainer.__call__()
    except:
      print("Job failed")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for ResNet50 FixRes",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning-rate', default=0.02, type=float, help='base learning rate')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--epochs', default=120, type=int, help='epochs')
    parser.add_argument('--batch', default=64, type=int, help='Batch by GPU')
    parser.add_argument('--node', default=1, type=int, help='GPU nodes')
    parser.add_argument('--workers', default=10, type=int, help='Numbers of CPUs')
    parser.add_argument('--imnet-path', default='/the/imagenet/path', type=str, help='ImageNet dataset path')
    parser.add_argument('--shared-folder-path', default='your/shared/folder', type=str, help='Shared Folder')
    parser.add_argument('--job-id', default='0', type=str, help='id of the execution')
    parser.add_argument('--local-rank', default=0, type=int, help='GPU: Local rank')
    parser.add_argument('--global-rank', default=0, type=int, help='GPU: glocal rank')
    parser.add_argument('--num-tasks', default=8, type=int, help='How many GPUs are used')
    args = parser.parse_args()
    run(args.input_size,args.learning_rate,args.epochs,args.batch,args.node,args.workers,args.imnet_path,args.shared_folder_path,args.job_id,args.local_rank,args.global_rank,args.num_tasks)
