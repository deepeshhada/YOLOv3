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


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def print_objects(boxes, class_names):    
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))

            
def plot_boxes(img, boxes, class_names, plot_labels, color = None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)
    
    width = img.shape[1]
    height = img.shape[0]
    fig, a = plt.subplots(1,1)
    a.imshow(img)

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(np.around((box[0] - box[2]/2.0) * width))
        y1 = int(np.around((box[1] - box[3]/2.0) * height))
        x2 = int(np.around((box[0] + box[2]/2.0) * width))
        y2 = int(np.around((box[1] + box[3]/2.0) * height))
        rgb = (1, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue  = get_color(0, offset, classes) / 255
            
            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color
        width_x = x2 - x1
        width_y = y1 - y2
        rect = patches.Rectangle(
			(x1, y2),
            width_x, width_y,
            linewidth = 2,
            edgecolor = rgb,
            facecolor = 'none'
		)

        a.add_patch(rect)
        if plot_labels:
            conf_tx = class_names[cls_id] + ': {:.1f}'.format(cls_conf)
            lxc = (img.shape[1] * 0.266) / 100
            lyc = (img.shape[0] * 1.180) / 100
            a.text(
                x1 + lxc, 
                y1 - lyc, 
                conf_tx, 
                fontsize = 24, 
                color = 'k',
                bbox = dict(facecolor = rgb, edgecolor = rgb, alpha = 0.8)
            )        
    plt.show()
