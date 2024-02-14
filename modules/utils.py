import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def intersection_over_union(bbox1,bbox2,box_format = 'midpoint'):
    '''
    Brief : 
    --------
    function to calculate intersection over union for 2 bboxes
    --------   
    Args:
    --------
    bbox1 : list of target bboxes in format [...,x1,y1,x2,y2] or [...,x_mid,y_mid,w,h] where '...' represent 
    batch number and the format depending on box_format argument where the first format is 'corners' and second 
    'midpoint' which is the default for yolo algorthim

    bbox2: list of predictions bboxes in format [...,x1,y1,x2,y2] or [...,x_mid,y_mid,w,h] where '...' represent 
    batch number and the format depending on box_format argument where the first format is 'corners' and second 
    'midpoint' which is the default for yolo algorthim

    box_format: format of given coordinates, values:['midpoint','corners']
    --------
    returns:
    --------
    iou: intersection aree is divided by union area
    --------
    '''

    if box_format == 'corners':
        box1_x1 = bbox1[...,0:1]
        box1_y1 = bbox1[...,1:2]
        box1_x2 = bbox1[...,2:3]
        box1_y2 = bbox1[...,3:4]

        box2_x1 = bbox2[...,0:1]
        box2_y1 = bbox2[...,1:2]
        box2_x2 = bbox2[...,2:3]
        box2_y2 = bbox2[...,3:4]
    elif box_format == 'midpoint':
        w1 = bbox1[...,2:3]
        h1 = bbox1[...,3:4]
        w2 = bbox2[...,2:3]
        h2 = bbox2[...,3:4]
        box1_x1 = bbox1[...,0:1] - w1/2
        box1_y1 = bbox1[...,1:2] - h1/2
        box1_x2 = bbox1[...,0:1] + w1/2
        box1_y2 = bbox1[...,1:2] + h1/2

        box2_x1 = bbox2[...,0:1] - w2/2
        box2_y1 = bbox2[...,1:2] - h2/2
        box2_x2 = bbox2[...,0:1] + w2/2
        box2_y2 = bbox2[...,1:2] + h2/2

    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1,box2_y1)
    x2 = torch.min(box1_x2,box2_x2)
    y2 = torch.min(box1_y2,box2_y2)

    intersection_area = (x2-x1).clamp(0)*(y2-y1).clamp(0)
    box1_area = abs((box1_x2-box1_x1) * (box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1) * (box2_y2-box2_y1))
    union_area = box1_area + box2_area - intersection_area 
    return intersection_area/union_area

def non_max_suppression(predictions,iou_th,prob_th,box_format="midpoint"):
    '''
    brief: 
    --------
    suppress most likely wrong bbox and keep only the appropiate ones
    --------
    Args:
    --------
    predictions: list of bboxes where every element in the list contains [class, pred_prob, x1,y1,x2,y2] or
    [class, pred_prob, x_mid,y_mid,w,h] depending on box_format value

    iou_th: threshold to suppress any bbox above has iou equal or greater than this threshold with chosen bbox

    prob_th: threshold to suppress any bbox below this threshold

    box_format: format of given coordinates, values:['midpoint','corners']
    --------
    returns:
    --------
    boxes_after_nms: list of appropiate bboxes
    --------
    '''
    boxes = [box for box in predictions if box[1]>prob_th]
    boxes = sorted(boxes,key=lambda x:x[1] , reverse=True)
    boxes_after_nms = []
    while boxes:
        chosen_box = boxes.pop(0)
        boxes = [box for box in boxes if box[0] != chosen_box[0] or 
                 intersection_over_union(torch.tensor(box[2:]),
                                         torch.tensor(chosen_box[2:]),
                                         box_format=box_format)
                                         <iou_th]
        boxes_after_nms.append(chosen_box)
    return boxes_after_nms

def mean_average_percision(pred_boxes,true_boxes,num_classes=20,iou_th=0.5,box_format='corners'):

    '''
    Brief:
    --------
    function to measure a metric called Mean Average Percision(MAP) which is simply the area under Percision-Reacll curve
    --------
    Args:
    --------
    pred_boxes: List of all predictions bboxes in the format [train_idx,class,probabilty_score,x1,y1,x2,y2]
    true_boxes: List of all true bboxes in the format [train_idx,class,probabilty_score,x1,y1,x2,y2]
    num_classes: number of classes in the data
    iou_th: iou threshold
    box_format: the bbox format, defalut is 'corners'
    --------
    return:
    --------
    map: mean average percision
    '''
    average_percision = []
    epislon = 1e-6
    for c in range(num_classes):
        detections = []
        ground_truth=[]
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1]==c:
                ground_truth.append(true_box)
        
        amount_bboxes = Counter([gt[0] for gt in ground_truth])

        for key,val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        #sorting according to probability score
        detections.sort(key=lambda x:x[2],reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truth)
        for detection_idx,detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truth if bbox[0]==detection[0]
            ]
            num_gth = len(ground_truth_img)
            best_iou = 0
            for idx,gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_th :
                if amount_bboxes[detection[0]][best_gt_idx]==0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP,dim=0)
        FP_cumsum = torch.cumsum(FP,dim=0)
        recall = TP_cumsum / (total_true_bboxes + epislon)
        recall = torch.cat((torch.tensor([0]),recall))
        percision = torch.divide(TP_cumsum,(TP_cumsum + FP_cumsum+ epislon))

        percision = torch.cat((torch.tensor([1]),percision))
        
        average_percision.append(torch.trapz(percision,recall))




    return sum(average_percision)/len(average_percision)



def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_th=iou_threshold,
                prob_th=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7,C=1):
   
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + 10)
    bboxes1 = predictions[..., C+1:C+5]
    bboxes2 = predictions[..., C+6:C+10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C+5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])