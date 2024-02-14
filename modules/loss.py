import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self,S=7 , B=2,C=20):
        super(YOLOLoss,self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self,prediction,targets):
        prediction = prediction.reshape(-1,self.S,self.S,self.C + 5 * self.B)
        iou_b1 = intersection_over_union(prediction[...,self.C+1:self.C+5],
                                        targets[...,self.C+1:self.C+5])
        iou_b2 = intersection_over_union(prediction[...,self.C+6:self.C+10],
                                        targets[...,self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)

        iou_maxes,best_box = torch.max(ious,dim=0)
        
        exists_box = targets[...,self.C].unsqueeze(3)  #Iobj_i

        # FOR BOX COORDS

        box_predictions = exists_box * (
            best_box * prediction[...,self.C+6:self.C+10]
            + (1-best_box) * prediction[...,self.C+1:self.C+5]
        )

        box_targets = exists_box * targets[...,self.C+1:self.C+5]


        # BOX LOSS

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) *torch.sqrt(
            torch.abs(box_predictions[...,2:4] + 1e-6)
        )
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])



        box_loss = self.mse(
            torch.flatten(box_predictions,end_dim=-2),
            torch.flatten(box_targets,end_dim=-2)
            )

        # OBJECT LOSS
        pred_box = (
            best_box * prediction[...,self.C+5:self.C+6]
            + (1-best_box) *prediction[...,self.C:self.C+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[...,self.C:self.C+1])
        )

        # NO OBJECT LOSS
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * prediction[...,self.C:self.C+1],start_dim=1),
            torch.flatten((1-exists_box) * targets[...,self.C:self.C+1],start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * prediction[...,self.C+5:self.C+6],start_dim=1),
            torch.flatten((1-exists_box) * targets[...,self.C:self.C+1],start_dim=1)
        )

        # FOR CLASS LOSS 

        class_loss = self.mse(
            torch.flatten(exists_box * prediction[...,:self.C],end_dim=-2),
            torch.flatten(exists_box * targets[...,:self.C],end_dim=-2)
        )

        loss =(
            self.lambda_coord *  box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss