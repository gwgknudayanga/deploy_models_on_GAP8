
import numpy as np
import cv2
import torch
import torchvision.transforms  as T
import torch.nn.functional as F
import random
import math

#None of the annotation arrays will not be modified inplace in the following functions

#################################### This function is only for evframe................

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    '''Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio.'''
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def random_affine(img, labels=(), degrees=10, translate=.1, scale=.1, shear=10,
                  new_shape=(448, 448)):
    '''Applies Random affine transformation.'''
    n = len(labels)
    if isinstance(new_shape, int):
        height = width = new_shape
    else:
        height, width = new_shape

    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    if n:
        new = np.zeros((n, 4))

        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        labels = labels[i]
        labels[:, 1:5] = new[i]

    return img, labels

#######################################################

def augment(event,anns):
    x_shift = 20
    y_shift = 10
    theta = 10
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)

    event[:,1] = event[:,1] * cos_theta - event[:,2] * sin_theta + xjitter
    event[:,2] = event[:,1] * sin_theta + event[:,2] * cos_theta + yjitter
    event[:,1] = np.clip(event[:,1],0,345)
    event[:,2] = np.clip(event[:,2],0,259)

    bboxes = anns[:,1:].copy()
    bboxes[:,[0,2]] = anns[:,[1,3]] * cos_theta - anns[:,[2,4]] * sin_theta + xjitter
    bboxes[:,[1,3]] = bboxes[:,[0,2]] * sin_theta + anns[:,[2,4]] * cos_theta + yjitter
    bboxes[:,[0,2]] = np.clip(bboxes[:,[0,2]],0,345)
    bboxes[:,[1,3]] = np.clip(bboxes[:,[1,3]],0,259)
    anns[:,1:] = bboxes
    
    return event,anns

def letterbox(im,anns,new_shape= 416):
    
    #Expected tensor input format --> TCHW
    #Expected annotation format --> (category_id,x_min,y_min,x_max,y_max)
    
    '''Resize and pad image while meeting stride-multiple constraints.'''
    im #this is a tensor
    h0,w0 = im.shape[-2:]

    ratio = new_shape / max(h0, w0)

    im = T.Resize((int(ratio*h0),int(ratio*w0)),T.InterpolationMode.NEAREST)(im)

    # Compute padding
    new_unpad = int(ratio*h0),int(ratio*w0)
    h,w = new_unpad
    dh, dw = new_shape - new_unpad[0], new_shape - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    pad2d = (left,right,top,bottom)  #(last,last,next_to_last,next_to_last)
    im = F.pad(im, pad2d, "constant", 0)   #Assuming im is in (T,X,H,W) format

    ratio = (h/h0,w/w0)
    pad = (top,left)

    bboxes = np.copy(anns[:, 1:])
    bboxes[:, 0] = ratio[1] * anns[:, 1] + pad[1] # top left x
    bboxes[:, 1] = ratio[0] * anns[:, 2] + pad[0] # top left y
    bboxes[:, 2] = ratio[1] * anns[:, 3] + pad[1] # bottom right x
    bboxes[:, 3] = ratio[0] * anns[:, 4] + pad[0] # bottom right y
    anns[:, 1:] = bboxes
    
    return im,anns,ratio,pad #Here we send the ratio and pad for future rescaling to original size 
                            #and then calculate metrics and visualization of detections

#vertical flip
def vflip(im,anns,img_height):
    
    boxes_cpy = anns[:,1:].copy()
    boxes_cpy[:,[1,3]] = img_height - anns[:,[2,4]]
    boxes_cpy[:,[1,3]] = boxes_cpy[:,[3,1]]

    if not isinstance(im, torch.Tensor):
        im = torch.from_numpy(im)
        
    anns[:,1:] = boxes_cpy
    return torch.flip(im,dims=[-2]),anns


def hflip(im,anns,img_width):

    boxes_cpy = anns[:,1:].copy()
    boxes_cpy[:,[0,2]] = img_width - anns[:,[1,3]]
    boxes_cpy[:,[0,2]] = boxes_cpy[:,[2,0]]

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)

    anns[:,1:] = boxes_cpy
    
    return torch.flip(im,dims=[-1]),anns

def rotate90antclk(im,anns,img_width):

    #rotating bboxes anticlockwise
    x_min,y_min,x_max,y_max = anns[:,1],anns[:,2],anns[:,3],anns[:,4]

    new_xmin = y_min
    new_ymin = img_width-x_max
    new_xmax = y_max
    new_ymax = img_width-x_min

    boxes_cpy = anns[:,1:].copy()

    boxes_cpy[:,0] = new_xmin
    boxes_cpy[:,1] = new_ymin
    boxes_cpy[:,2] = new_xmax
    boxes_cpy[:,3] = new_ymax

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)
    anns[:,1:] = boxes_cpy
    
    return torch.rot90(im,k = 1,dims=[-2,-1]),anns #rotate anticlockwise

def rotate90clk(im,anns,img_height):
    x_min,y_min,x_max,y_max = anns[:,1],anns[:,2],anns[:,3],anns[:,4]

    new_xmin = img_height - y_max
    new_ymin = x_min
    new_xmax = img_height - y_min
    new_ymax = x_max

    boxes_cpy = anns[:,1:].copy()

    boxes_cpy[:,0] = new_xmin
    boxes_cpy[:,1] = new_ymin
    boxes_cpy[:,2] = new_xmax
    boxes_cpy[:,3] = new_ymax

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)

    anns[:,1:] = boxes_cpy
    return torch.rot90(im,k = -1,dims=[-2,-1]),anns

def randomcrop(im,anns,img_width,img_height):

    #This has to called padding and resize

    crop_length_list = [150,150]
    crop_length = crop_length_list[0]
    
    max_x = img_width - crop_length
    max_y = img_height - crop_length

    start_x = np.random.randint(0, max_x + 1)
    start_y = np.random.randint(0, max_y + 1)

    cropped_image = im[:,start_y:start_y + crop_length, start_x:start_x + crop_length]

    temp_boxes = anns[:,1:].copy()

    temp_boxes[:,[0,2]] -= start_x
    temp_boxes[:,[1,3]] -= start_y
    
    temp_boxes[:,[0,2]] = np.clip(temp_boxes[:,[0,2]],0,crop_length)
    temp_boxes[:,[1,3]] = np.clip(temp_boxes[:,[1,3]],0,crop_length)

    rect_1 = (start_x,start_y,start_x + crop_length, start_y + crop_length)
    rect_2 = (temp_boxes[:,0],temp_boxes[:,1],temp_boxes[:,2],temp_boxes[:,3])

    """is_sufficient_intersecting_rectangle(rect_1,rect_2):
        print("sufficient rectangle found ")
        anns[:,1:] = temp_boxes
        return cropped_image,anns"""
    
    anns[:,1:] = temp_boxes
    return cropped_image,anns

def is_sufficient_intersecting_rectangle(rect1, rect2):
    """
    Check if two rectangles intersect and return the intersecting rectangle if they do.
    
    Each rectangle is represented as a tuple of four values:
    (x1, y1, x2, y2)
    where (x1, y1) are the coordinates of the bottom-left corner and
    (x2, y2) are the coordinates of the top-right corner.
    
    :param rect1: Tuple (x1, y1, x2, y2) for the first rectangle
    :param rect2: Tuple (x1, y1, x2, y2) for the second rectangle
    :return: Tuple (x1, y1, x2, y2) for the intersecting rectangle if they intersect, 
             None otherwise
    """
    
    # Unpack the rectangle coordinates
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    
    # Check if the rectangles intersect
    if x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1:
        return False
    
    # Calculate the coordinates of the intersecting rectangle
    inter_left = max(x1, x3)
    inter_bottom = max(y1, y3)
    inter_right = min(x2, x4)
    inter_top = min(y2, y4)
    width = inter_right - inter_left
    height = inter_top - inter_bottom

    if width < 5 or height < 5:
        return False