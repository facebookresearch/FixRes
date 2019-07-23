# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .inception import inceptionresnetv2

import torch
import torch.distributed
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch.nn as nn
from torchvision import datasets
from .transforms import get_transforms
import pandas as pd
from .senet import se_resnext101_32x4d,senet154
from .RedefinedModel_MPN import resnet152_MPN_COV# We don't provide this code


def Run(data_Folder,num_workers,batch_per_gpu,indice_path,output_path,senet_path,resnet_path,inception_path,seresnext_path):

    transf=get_transforms(input_size=704,test_size=704, kind='full', crop=True, need=('train', 'val'), backbone=None)
    
    
    test_set=datasets.ImageFolder(data_Folder+'/test',transform=transf['val'])
    val_set=datasets.ImageFolder(data_Folder + '/val',transform=transf['val'])

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_per_gpu, shuffle=False, num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_per_gpu, shuffle=False, num_workers=num_workers,
    )
    transf2=get_transforms(input_size=681,test_size=681, kind='full', crop=True, need=('train', 'val'), backbone='pnasnet5large')
    
    
    test_set2=datasets.ImageFolder(data_Folder+'/test',transform=transf2['val'])
    val_set2=datasets.ImageFolder(data_Folder + '/val',transform=transf2['val'])
    


    val_loader2 = torch.utils.data.DataLoader(
        val_set2, batch_size=batch_per_gpu, shuffle=False, num_workers=num_workers,
    )

    test_loader2 = torch.utils.data.DataLoader(
        test_set2, batch_size=batch_per_gpu, shuffle=False, num_workers=9,
    )
    transf3=get_transforms(input_size=672,test_size=672, kind='full', crop=True, need=('train', 'val'), backbone=None)
    
    
    test_set3=datasets.ImageFolder(data_Folder+'/test',transform=transf3['val'])
    val_set3=datasets.ImageFolder(data_Folder + '/val',transform=transf3['val'])
    


    val_loader3 = torch.utils.data.DataLoader(
        val_set3, batch_size=batch_per_gpu, shuffle=False, num_workers=num_workers,
    )


    test_loader3 = torch.utils.data.DataLoader(
        test_set3, batch_size=batch_per_gpu, shuffle=False, num_workers=num_workers,
    )
    transf4=get_transforms(input_size=480,test_size=480, kind='full', crop=True, need=('train', 'val'), backbone=None)
    
    test_set4=datasets.ImageFolder(data_Folder+'/test',transform=transf4['val'])
    val_set4=datasets.ImageFolder(data_Folder + '/val',transform=transf4['val'])
    

    val_loader4 = torch.utils.data.DataLoader(
        val_set4, batch_size=batch_per_gpu, shuffle=False, num_workers=num_workers,
    )


    test_loader4 = torch.utils.data.DataLoader(
        test_set4, batch_size=batch_per_gpu, shuffle=False, num_workers=num_workers, 
    )


    
    model = se_resnext101_32x4d(pretrained='imagenet')
    model.last_linear=nn.Linear(2048, 1010)

    pretrained_dict=torch.load(seresnext_path,map_location='cpu')['model']
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

    
    model2 = inceptionresnetv2(pretrained='imagenet')
    model2.last_linear=nn.Linear(1536, 1010)

    pretrained_dict=torch.load(inception_path,map_location='cpu')['model']
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

    
    model3 = senet154(pretrained='imagenet')
    model3.last_linear=nn.Linear(2048, 1010)

    pretrained_dict=torch.load(senet_path,map_location='cpu')['model']
    model_dict = model3.state_dict()
    count=0
    count2=0
    for k in model_dict.keys():
        count=count+1.0
        if(('module.'+k) in pretrained_dict.keys()):
            count2=count2+1.0
            model_dict[k]=pretrained_dict.get(('module.'+k))
    model3.load_state_dict(model_dict)
    print("load "+str(count2*100/count)+" %")
    for name, child in model3.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = False
    
    model3.cuda()

    
    model4 = resnet152_MPN_COV(pretrained=True)
    model4.fc=nn.Linear(int(256*(256+1)/2), 1010)

    pretrained_dict=torch.load(resnet_path,map_location='cpu')['model']
    model_dict = model4.state_dict()
    count=0
    count2=0
    for k in model_dict.keys():
        count=count+1.0
        if(('module.'+k) in pretrained_dict.keys()):
            count2=count2+1.0
            model_dict[k]=pretrained_dict.get(('module.'+k))
    model4.load_state_dict(model_dict)
    print("load "+str(count2*100/count)+" %")
    for name, child in model4.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = False
    
    model4.cuda()




    criterion = nn.CrossEntropyLoss()
    acc = None

    
    correct = 0
    total = 0
    count=0.0
    running_val_loss = 0.0
    model2.eval()
    model3.eval()
    model.eval()
    model4.eval()
    with torch.no_grad():
        for data,data2,data3,data4 in zip(val_loader,val_loader2,val_loader3,val_loader4):
            
            images, labels = data
            bs, ncrops, c, h, w = images.size()

            labels = labels.cuda()
            result = model(images.view(-1, c, h, w).cuda())
            images4, labels = data4
            bs, ncrops, c, h, w = images4.size()

            labels = labels.cuda()
            result4 = model4(images4.view(-1, c, h, w).cuda())

            images2, labels = data2
            bs, ncrops, c, h, w = images2.size()

            labels = labels.cuda()
            result2 = model2(images2.view(-1, c, h, w).cuda())
            
            images3, labels3 = data3
            bs, ncrops, c, h, w = images3.size()

            labels3 = labels3.cuda()
            result3 = model3(images3.view(-1, c, h, w).cuda())
            outputs = (result4.view(bs, ncrops, -1).mean(1)  +result.view(bs, ncrops, -1).mean(1)  +result3.view(bs, ncrops, -1).mean(1)  +result2.view(bs, ncrops, -1).mean(1)  )/4.0
            loss_val = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_val_loss += loss_val.item()
            count=count+1.0

    acc = correct / total
    ls_nm=running_val_loss/count
    print(f"Accuracy of the network  {acc:.1%}", flush=True)
    print(f"Loss of the network  {ls_nm:.3f}", flush=True)

    
    transf=get_transforms(input_size=704,test_size=704, kind='full', crop=True, need=('train', 'val'), backbone=None)
    transform_test = transf['val']
    train_set = datasets.ImageFolder(data_Folder + '/train',transform=transform_test)
    test_set = datasets.ImageFolder(data_Folder + '/test',transform=transform_test)

    
    correct = 0
    total = 0
    count=0.0
    running_val_loss = 0.0
    df=pd.read_csv(indice_path+'test_info.csv')
    id_test=[]
    pred_test=[]
    model2.eval()
    model3.eval()
    model.eval()
    model4.eval()
    with torch.no_grad():
        for data,data2,data3,data4 in zip(test_loader,test_loader2,test_loader3,test_loader4):
          
            images, labels = data
            bs, ncrops, c, h, w = images.size()

            labels = labels.cuda()
            result = model(images.view(-1, c, h, w).cuda())
            images4, labels = data4
            bs, ncrops, c, h, w = images4.size()

            labels = labels.cuda()
            result4 = model4(images4.view(-1, c, h, w).cuda())
            images2, labels = data2
            bs, ncrops, c, h, w = images2.size()

            labels = labels.cuda()
            result2 = model2(images2.view(-1, c, h, w).cuda())
            
            images3, labels3 = data3
            bs, ncrops, c, h, w = images3.size()

            labels3 = labels3.cuda()
            result3 = model3(images3.view(-1, c, h, w).cuda())
            
            
            outputs = ( result4.view(bs, ncrops, -1).mean(1) +result.view(bs, ncrops, -1).mean(1) +result2.view(bs, ncrops, -1).mean(1) + result3.view(bs, ncrops, -1).mean(1))/4.0               
            _, predicted = torch.max(outputs.data, 1)
            id_test.extend([int(train_set.classes[i]) for i in predicted.data.cpu().numpy()])
            pred_test.extend([df.loc[df["indice"]==int(test_set.classes[i])]["id"].values[0] for i in labels.data.cpu().numpy()])


    
    data={"Id":pred_test,"Predicted":id_test}
    dt=pd.DataFrame(data)
    dt.to_csv(output_path+'submission.csv',index=False)
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
    parser.add_argument('--inception-path', default='', type=int, help='InceptionResNetV2 pth file')
    parser.add_argument('--seresnext-path', default='', type=int, help='SEResNext pth file')

    args = parser.parse_args()
    Run(args.data_Folder,args.num_workers,args.batch_per_gpu,args.indice_path,args.output_path,args.senet_path,args.resnet_pathargs.inception_path,args.seresnext_path)

