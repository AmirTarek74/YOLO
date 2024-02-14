import os
import torch
from torchvision.io import read_image
from PIL import Image
import cv2


class Data(torch.utils.data.Dataset):
  def __init__(self,dir,transform=None,S=7,B=2,C=1):
    self.dir = dir
    self.imgs = os.listdir(os.path.join(dir,'images'))
    self.labels = os.listdir(os.path.join(dir,'labels'))
    self.transform = transform
    self.S = S
    self.B = B
    self.C = C

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self,index):
    boxes = []
    label = os.path.join(self.dir,'labels/'+self.labels[index])
    with open(label) as f:
      for line in f.readlines():
        class_label,x,y,w,h = [
              float(x) if float(x) != int(float(x)) else int(x)
              for x in line.replace('\n',"").split()
          ]
        boxes.append([class_label,x,y,w,h])
    image = Image.open(os.path.join(self.dir,'images/'+self.imgs[index]))
    boxes = torch.tensor(boxes)

    if self.transform:
      image,boxes = self.transform(image,boxes)

    label_matrix = torch.zeros((self.S,self.S,self.C + self.B * 5))
    
    for box in boxes:
      class_label , x,y,w,h = box.tolist()
      class_label = int(class_label)
      i,j = int(self.S * y), int(self.S * x)
      x_cell, y_cell = self.S * x - j , self.S * y - i
      width_cell,height_cell=(
          w * self.S,
          h * self.S,
      )

      if label_matrix[i,j,self.C]==0:   #if there is no object in this i j cell
        label_matrix[i,j,self.C] = 1
        box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
        label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
        label_matrix[i, j, class_label] = 1

    return image,label_matrix