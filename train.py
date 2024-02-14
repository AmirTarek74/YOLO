import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
import tqdm as tqdm
import numpy as np 
import os 
import cv2
from torch.utils.data import Dataset,DataLoader
from model import YOLOv1
from loss import YOLOLoss
from utils import *
from data_loader import Data


seed = 42
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE ='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = os.cpu_count()
INPUT_SHAPE =(224,224)
DATA_DIR = "./data"
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 3


class Compose(object):
    def __init__(self,transforms) :
        self.transforms = transforms
    
    def __call__(self,img,bboxes):
        for t in self.transforms:
            img,bboxes = t(img),bboxes
        
        return img,bboxes
    
transform = Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.ToTensor()
])

def train_fun(train_loader,model,optimizer,loss_fun):
    mean_loss = []
    for batch,(img,labels) in enumerate(train_loader):
        img,labels = img.to(DEVICE),labels.to(DEVICE)
        output = model(img)
        loss = loss_fun(output,labels)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
    print(f"Mean Loss : {sum(mean_loss)/len(mean_loss)}")

def Val_fun(val_loader,model,optimizer,loss_fun):
    mean_loss = []
    model.eval()
    with torch.inference_mode():
        for batch,(img,labels) in enumerate(val_loader):
            img,labels = img.to(DEVICE),labels.to(DEVICE)
            output = model(img)
            loss = loss_fun(output,labels)
            mean_loss.append(loss.item())

        print(f"Valdiation Mean Loss : {sum(mean_loss)/len(mean_loss)}")

directeries = ['train','test','valid']
for directory in directeries:
    path = os.path.join(DATA_DIR,directory)
    if directory =='train':
        train_data = Data(path,transform=transform,S=SPLIT_SIZE,B=NUM_BOXES,C=NUM_CLASSES)
    elif directory =='test':
        test_data = Data(path,transform=transform,S=SPLIT_SIZE,B=NUM_BOXES,C=NUM_CLASSES)
    else:
        valid_data = Data(path,transform=transform,S=SPLIT_SIZE,B=NUM_BOXES,C=NUM_CLASSES)

train_loader = DataLoader(train_data,batch_size = BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=True,drop_last=True)
test_loader = DataLoader(test_data,batch_size = BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=True,drop_last=True)
valid_loader = DataLoader(valid_data,batch_size = BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=True,drop_last=True)

model = YOLOv1(split_size = SPLIT_SIZE, num_boxes=NUM_BOXES,num_classes=NUM_CLASSES).to(DEVICE)
loss_fn = YOLOLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

for epoch in range(EPOCHS):
    print(f"Epoch{epoch+1}/{EPOCHS}\n--------")
    pred_boxes , target_boxes = get_bboxes(train_loader,model,iou_threshold=0.5,threshold=0.4)
    mean_avg_pred = mean_average_percision(pred_boxes,target_boxes,iou_th=0.5,box_format='midpoint')
    print(f"Train Map :{mean_avg_pred:.4f}")
    
    train_fun(train_loader,model,optimizer,loss_fn)
    Val_fun(valid_loader,model,optimizer,loss_fn)

