import os
import torch
import numpy as np
import pandas as pd
from torch import nn
import albumentations as A
import cv2
import torchsummary
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

torch.cuda.empty_cache()

BS = 4
IS = 256
D = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 7
classes=['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
c2l={k : v for k, v in list(zip(classes,list(range(num_classes))))}
l2c={v : k for k, v in c2l.items()}

dir_path = 'Dataset_Fracture_Detection'
train_dir_path = 'Dataset_Fracture_Detection/train'
train_img_paths = sorted(os.listdir('Dataset_Fracture_Detection/train/images'))
train_target_paths = sorted(os.listdir('Dataset_Fracture_Detection/train/labels'))

val_dir_path = 'Dataset_Fracture_Detection/valid'
val_img_paths = sorted(os.listdir('Dataset_Fracture_Detection/valid/images'))
val_target_paths = sorted(os.listdir('Dataset_Fracture_Detection/valid/labels'))

#Function to resize boundary boxes wrt the image transformation sizes
def unconvert(width, height, x, y, w, h):

    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)

    return xmin, ymin, xmax, ymax

#print and check with boundary boxes

idx = random.randint(0,3000)

ip = os.path.join(train_dir_path, 'images', train_img_paths[idx])
tp = os.path.join(train_dir_path, 'labels', train_target_paths[idx])

image = cv2.imread(ip)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (320, 320))

file = open(tp, 'r')
target = list(map(float, file.read().split()))[1:]

a = 0
while a < len(target):
    bbox = target[a : a+4] #plot each box
    if len(bbox) == 4:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin, ymin, xmax, ymax = unconvert(320, 320, x, y, w, h)
        sp, ep = (xmin, ymin),(xmax, ymax)
        image = cv2.rectangle(image, sp, ep, (255, 0, 0), 2)
    a += 4

plt.imshow(image)
plt.show()

augs = A.Compose([
    A.Resize(IS, IS),
],bbox_params = A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels']), is_check_shapes = True)

class FractureData(torch.utils.data.Dataset):

    def __init__(self, dir_path, img_paths, target_paths, augs = None):
        self.dir_path = dir_path
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.augs = augs

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        ip = os.path.join(self.dir_path,'images',self.img_paths[idx])
        tp = os.path.join(self.dir_path,'labels',self.target_paths[idx])

        image =cv2.imread(ip)
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        file = open(tp, 'r')
        target = list(map(float, file.read().split()))

        try:
            label = [target.pop(0)]
            bbox = []
            i = 0
            while i < len(target):
                x, y, w, h = target[i : i + 4]
                bbox.append([*unconvert(W, H, x, y, w, h)])
                i += 4
            label = label * len(bbox)

            if self.augs != None:
                data = self.augs(image = image, bboxes = bbox, class_labels = ['None'] * len(label))
                image = data['image']
                bbox = data['bboxes']
        except:
            if idx + 1 < len(self.img_paths):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)

        image = torch.Tensor(np.transpose(image, (2, 0, 1))) / 255.0
        bbox = torch.Tensor(bbox).long()
        label = torch.Tensor(label).long()

        annot = {'boxes' : bbox, 'labels' : label}

        return image, annot

    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
trainset = FractureData(train_dir_path, train_img_paths, train_target_paths, augs)
valset = FractureData(val_dir_path, val_img_paths, val_target_paths, augs)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = BS, collate_fn = trainset.collate_fn)
valloader = torch.utils.data.DataLoader(valset, batch_size = BS, collate_fn = valset.collate_fn)

print(f'Training Data:- {len(trainset)} images divided into {len(trainloader)} batches')

for image,target in trainloader:
    break

#plot image with boundary boxes and labels

idx = random.randint(0, len(image) -1)
img, tar = image[idx].permute(1, 2, 0).numpy(), target[idx]

for bbox in tar['boxes']:
    xmin, ymin, xmax, ymax = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    sp, ep = (xmin, ymin), (xmax, ymax)
    cv2.rectangle(img, sp, ep, (0, 255, 0), 1)

plt.imshow(img)
plt.title(l2c[tar['labels'][0].item()])
plt.show()

#load the pre-trained model

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model=torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# model.load_state_dict(torch.load('model/Resnet.pt', map_location = D))
# model.eval()
model.to(D)

#define train model
def trainarc(model, dataloader, opt):
    model.train()
    train_loss = 0.0

    for images, targets in tqdm(dataloader):
        image=[i.to(D) for i in images]
        target=[{k : v.to(D) for k, v in ele.items()} for ele in targets]

        opt.zero_grad()
        losses = model(image, target)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        opt.step()

        train_loss += loss
    return train_loss / len(dataloader)

#define validate model
def evalarc(model, dataloader):
    model.train()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            image = [i.to(D) for i in images]
            target = [{k : v.to(D) for k, v in ele.items()} for ele in targets]

            losses = model(image, target)
            loss = sum( loss for loss in losses.values() )

            val_loss += loss
    return val_loss / len(dataloader)

#train the model

LR = 0.00003
epochs = 30

best_val_loss = np.inf

opt = torch.optim.Adam(model.parameters(),lr = LR)

for i in range(epochs):
    train_loss = trainarc(model, trainloader, opt)
    val_loss = evalarc(model, valloader)

    print(f"Epochs: {i + 1}/{epochs}:- Trainloss: {train_loss}, Valloss: {val_loss}")

    if val_loss < best_val_loss:
        torch.save(model.state_dict(),'model/model.pt')
        print("Model Updated")
        best_val_loss = val_loss

torch.save(model.state_dict(),'model/Resnet.pt') # Model Name
print("Fully Trained Model Saved")
print(f"Done. Best Val Loss: {best_val_loss}")