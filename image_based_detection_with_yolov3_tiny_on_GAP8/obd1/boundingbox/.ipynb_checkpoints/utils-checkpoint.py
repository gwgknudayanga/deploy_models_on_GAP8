# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as Img
from PIL import ImageDraw
from PIL.Image import Image
from torchvision import ops, transforms

from .metrics import bbox_iou

"""Utility functions for Object detection."""

RGB = int
Width = int
Height = int

def tensor_from_annotation(ann: Dict[str, Any],
                           device: torch.device = torch.device('cpu'),
                           num_objects: Optional[int] = None,
                           normalized: bool = True) -> torch.tensor:
    """Translate annotation dictionary to bounding box. This is not compatible
    with batch processing.

    Parameters
    ----------
    ann : Dict[str, Any]
        Object annotation dictionary with objects in the format of
        ``{'id':id, 'name':name,
        'confidence':conf,
        'bndbox':{'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}}``.
    device : torch.device, optional
        Target torch backend, by default torch.device('cpu').
    num_objects : Optional[int], optional
        Maximum number of objects to return. If None, all of them are
        translated. By default None.
    normalized : bool, optional
        Flag indicating use of normalized annotation format with coordinates
        between 0 and 1 in the tensor output, by default True.

    Returns
    -------
    torch.tensor
        Annotation tensor. Every column is an object. The rows are in the order
        of x_center, y_cener, width, height, confidence(=1), label_id.
    """
    if normalized:
        height = int(ann['annotation']['size']['height'])
        width = int(ann['annotation']['size']['width'])
    else:
        height = 1
        width = 1
    boxes = []
    area = []
    for object in ann['annotation']['object']:
        if 'confidence' in object.keys():
            confidence = object['confidence']
        else:
            confidence = 1.0
        xmin = object['bndbox']['xmin']
        xmax = object['bndbox']['xmax']
        ymin = object['bndbox']['ymin']
        ymax = object['bndbox']['ymax']
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([(xmin + xmax) / width / 2,
                      (ymin + ymax) / height / 2,
                      (xmax - xmin) / width,
                      (ymax - ymin) / height,
                      confidence,  # confidence
                      object['id']])  # label
        area.append((xmax - xmin) * (ymax - ymin))

    idx = np.argsort(area)[::-1]
    boxes = torch.FloatTensor([boxes[i] for i in idx], device=device)
    if num_objects:
        if num_objects < len(boxes):
            boxes = boxes[:num_objects]

    return boxes

def non_maximum_suppression(predictions: List[torch.tensor],
                            conf_threshold: float = 0.02,
                            nms_threshold: float = 0.4,
                            merge_conf: bool = True,
                            max_detections: int = 300,
                            max_iterations: int = 100) -> List[torch.tensor]:
    """Performs Non-Maximal suppression of the input predictions. First a basic
    filtering of the bounding boxes based on a minimum confidence threshold are
    eliminated. Subsequently a non-maximal suppression is performed. A
    non-maximal threshold is used to determine if the two bounding boxes
    represent the same object. It supports batch inputs.

    Parameters
    ----------
    predictions : List[torch.tensor]
        List of bounding box predictions per batch in
        (x_center, y_center, width, height) format.
    conf_threshold : float, optional
        Confidence threshold, by default 0.5.
    nms_threshold : float, optional
        Non maximal overlap threshold, by default 0.4.
    merge_conf : bool, optional
        Flag indicating whether to merge objectness score with classification
        confidence, by default True.
    max_detections : int, optional
        Maximum limit of detections to reduce computational load. If exceeded
        only the top predictions are taken., by default 300.
    max_iterations : int, optional
        Maximum number of iterations in non-maximal suppression loop, by
        default 100.

    Returns
    -------
    List[torch.tensor]
        Non-maximal filterered prediction outputs per batch.
    """
    result = []
    for pred in predictions:
        filtered = pred[pred[:, 4] > conf_threshold]
        if not filtered.size(0):
            result.append(torch.zeros((0, 6), device=pred.device))
            continue

        boxes = filtered[:, :4]
        obj_conf, labels = torch.max(filtered[:, 5:], dim=1, keepdim=True)
        if merge_conf:
            scores = filtered[:, 4:5] * obj_conf
        else:
            scores = filtered[:, 4:5]

        order = torch.argsort(scores.squeeze(), descending=True)
        # Custon NMS loop
        detections = torch.cat([boxes, scores, labels], dim=-1)
        prev_objects = detections.shape[0]
        if order.shape:
            detections = detections[order]
            for i in range(max_iterations):
                ious = bbox_iou(detections, detections)
                label_match = (
                    detections[:, 5].reshape(-1, 1)
                    == detections[:, 5].reshape(1, -1)
                ).long().view(ious.shape)

                keep = (
                    ious * label_match > nms_threshold
                ).long().triu(1).sum(dim=0,
                                     keepdim=True).T.expand_as(detections) == 0

                detections = detections[keep].reshape(-1, 6).contiguous()
                if detections.shape[0] == prev_objects:
                    break
                else:
                    prev_objects = detections.shape[0]
        # #
        # # above gives slightly better scores
        # idx = ops.nms(ops._box_convert._box_xywh_to_xyxy(boxes.clone()),
        #               scores.flatten(), nms_threshold)
        # detections = torch.cat([boxes[idx], scores[idx], labels[idx]], dim=-1)

        if detections.shape[0] > max_detections:
            detections = detections[:max_detections]
        result.append(detections)
    return result

nms = non_maximum_suppression