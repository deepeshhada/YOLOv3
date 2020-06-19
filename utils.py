import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def boxes_iou(box1, box2):
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    mx = min(box1[0] - width_box1 / 2.0, box2[0] - width_box2 / 2.0)
    Mx = max(box1[0] + width_box1 / 2.0, box2[0] + width_box2 / 2.0)
    my = min(box1[1] - height_box1 / 2.0, box2[1] - height_box2 / 2.0)
    My = max(box1[1] + height_box1 / 2.0, box2[1] + height_box2 / 2.0)
    union_width = Mx - mx
    union_height = My - my

    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    intersection_area = intersection_width * intersection_height
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area

    return iou


def nms(boxes, iou_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]
    _, sortIds = torch.sort(det_confs, descending=True)

    best_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            best_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0

    return best_boxes


def detect_objects(model, img, iou_thresh, nms_thresh):
    start = time.time()
    model.eval()
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    list_boxes = model(img, nms_thresh)
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    boxes = nms(boxes, iou_thresh)
    finish = time.time()

    print('\n\nIt took {:.3f}'.format(finish - start),
          'seconds to detect the objects in the image.\n')

    print('Number of Objects Detected:', len(boxes), '\n')
    return boxes
