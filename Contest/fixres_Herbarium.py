# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.distributed
import torch.nn as nn
from torchvision import datasets
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torchvision.models as models
from .transforms import get_transforms
from .senet import senet154

def Run(data_Folder,num_workers,batch_per_gpu,indice_path,output_path,senet_path,resnet_path):

    
    transf=get_transforms(input_size=707,test_size=707, kind='full', crop=True, need=('train', 'val'), backbone=None)
    transform_test = transf['val']
    
   
    val_set = datasets.ImageFolder(data_Folder + '/validation',transform=transform_test)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_per_gpu, shuffle=False, num_workers=9,
    )
    
    test_set = datasets.ImageFolder(data_Folder + '/test',transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_per_gpu, shuffle=False, num_workers=9,
    )
    transf2=get_transforms(input_size=640,test_size=640, kind='full', crop=True, need=('train', 'val'), backbone=None)
    transform_test2 = transf2['val']
    
   
    val_set2 = datasets.ImageFolder(data_Folder + '/validation',transform=transform_test2)

    val_loader2 = torch.utils.data.DataLoader(
        val_set2, batch_size=batch_per_gpu, shuffle=False, num_workers=9,
    )
    
    test_set2 = datasets.ImageFolder(data_Folder + '/test',transform=transform_test2)

    test_loader2 = torch.utils.data.DataLoader(
        test_set2, batch_size=batch_per_gpu, shuffle=False, num_workers=9,
    )


    
    model = senet154(pretrained='imagenet')
    model.last_linear=nn.Linear(2048, 683)
    pretrained_dict=torch.load(senet_path,map_location='cpu')['model']
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
    for name, child in model.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = False
    
    model.cuda()

    
    model2 = models.resnet50(pretrained=False)
    model2.fc=nn.Linear(2048,683)
    pretrained_dict=torch.load(resnet_path,map_location='cpu')['model']
    model_dict = model2.state_dict()
    count=0
    count2=0
    for k in model_dict.keys():
        count=count+1.0
        if(('module.'+k) in pretrained_dict.keys()):
            count2=count2+1.0
            model_dict[k]=pretrained_dict.get(('module.'+k))
    model2.load_state_dict(model_dict)
    print("load "+str(count2*100/count)+" %")
    
    for name, child in model2.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = False
    
    model2.cuda()


    print("Start evaluation of the model")
    
    correct = 0
    total = 0
    count=0.0
    model.eval()
    model2.eval()
    
    with torch.no_grad():
        for data,data2 in zip(val_loader,val_loader2):
            images, labels = data
            del data
            bs, ncrops, c, h, w = images.size()
            labels = labels.cuda()
            result1 = model(images.view(-1, c, h, w).cuda())
            del images
            images, _ = data2
            bs, ncrops, c, h, w = images.size()
            result2 = model2(images.view(-1, c, h, w).cuda())
            
            outputs = (result1.view(bs, ncrops, -1).mean(1)  +result2.view(bs, ncrops, -1).mean(1))/2.0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count=count+1.0
            if count%10==0:
                print(count)

    acc = correct / total
    print("Accuracy of the network on the validation images: {acc:.1%}")
    
    transf=get_transforms(input_size=707,test_size=707, kind='full', crop=True, need=('train', 'val'), backbone=None)
    transform_test = transf['val']
    train_set = datasets.ImageFolder(data_Folder + '/train',transform=transform_test)
    test_set = datasets.ImageFolder(data_Folder + '/test',transform=transform_test)
    correct = 0
    total = 0
    count=0.0
    df=pd.read_csv(indice_path+'test_cp.csv')
    id_test=[]
    pred_test=[]
    model.eval()
    model2.eval()
    with torch.no_grad():
        for data,data2 in zip(test_loader,test_loader2):
            images, labels = data
            del data
            bs, ncrops, c, h, w = images.size()
            labels = labels.cuda()
            result1 = model(images.view(-1, c, h, w).cuda())
            del images
            images, _ = data2
            bs, ncrops, c, h, w = images.size()
            result2 = model2(images.view(-1, c, h, w).cuda())
            count=count+1
            outputs = (result1.view(bs, ncrops, -1).mean(1)  +result2.view(bs, ncrops, -1).mean(1))/2.0               
            _, predicted = torch.max(outputs.data, 1)
            id_test.extend([int(train_set.classes[i]) for i in predicted.data.cpu().numpy()])
            pred_test.extend([df.loc[df["indice"]==int(test_set.classes[i])]["id"].values[0] for i in labels.data.cpu().numpy()])


    
    data={"Id":pred_test,"Category":id_test}
    dt=pd.DataFrame(data)
    dt.to_csv(output_path+'/submission.csv',index=False)
    return acc

if __name__ == "__main__":
    parser = ArgumentParser(description="script for FixRes Herbarium",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-Folder', default='', type=str, help='dataset path')
    parser.add_argument('--num-workers', default=0, type=int, help='CPUs')
    parser.add_argument('--batch-per-gpu', default=64, type=int, help='Batch per GPUs')
    parser.add_argument('--indice-path', default='', type=str, help='Path for indice file')
    parser.add_argument('--output-path', default='', type=str, help='where outputs are stored')
    parser.add_argument('--senet-path', default='', type=str, help='SENet pth file')
    parser.add_argument('--resnet-path', default='', type=int, help='ResNet pth file')

    args = parser.parse_args()
    Run(args.data_Folder,args.num_workers,args.batch_per_gpu,args.indice_path,args.output_path,args.senet_path,args.resnet_path)

