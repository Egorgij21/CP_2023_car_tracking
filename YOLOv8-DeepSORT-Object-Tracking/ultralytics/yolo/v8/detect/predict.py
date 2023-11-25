# Ultralytics YOLO 🚀, GPL-3.0 license

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from numpy import random

print(f"CUDA ISSSSSSS {torch.cuda.is_available()}")
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../../../')
# Ultralytics YOLO 🚀, GPL-3.0 license

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from numpy import random

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}

object_counter1 = {}

line = [(100, 500), (1050, 500)]
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=False)
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        return Annotator(img, line_width=self.args.line_thickness, example=str(names))

    def preprocess(self, img):
        img = torch.from_numpy(img)
        # .to(self.model.device)
        img = img.half() 
        # if self.model.fp16 else img.float()  # uint8 to fp16/32
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds
    
    
    def merge_areas(self, areas):
        """ Merge multiple areas into a single area using cv2 for convex hull calculation.

        Args:
        areas (list): A list of areas, where each area is represented as a list of points.

        Returns:
        np.ndarray: An array of points representing the merged area.
        """
        # Combine all points from all areas into a single array
        all_points = np.vstack(areas).reshape(-1, 2)

        # Calculate the convex hull
        hull_indices = cv2.convexHull(np.array(all_points).astype(np.float32), returnPoints=False)
        merged_area = all_points[hull_indices.squeeze()]

        return merged_area

    def is_point_in_area(self, point):
        """ Check if a point is inside a quadrilateral area using numpy for optimization.

        Args:
        point (tuple): A tuple representing the point (x, y).
        area (np.ndarray): A numpy array of points representing the area.

        Returns:
        bool: True if the point is inside the area, False otherwise.
        """
        area = self.area
        
        x_coords, y_coords = area[:, 0], area[:, 1]

        n = len(area)
        inside = 0

        for i in range(n):
            j = (i + 1) % n
            if ((y_coords[i] > point[1]) != (y_coords[j] > point[1])) and (point[0] < (x_coords[j] - x_coords[i]) * (point[1] - y_coords[i]) / (y_coords[j] - y_coords[i]) + x_coords[i]):
                inside = 1
        # print(inside)
        return inside
    
    def get_preds(self, idx, preds, batch):
        # start =time.time()
        p, im, im0 = batch
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        
        frame = getattr(self.dataset, 'frame', 0)
        self.annotator = self.get_annotator(im0)
        det = preds[idx]
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
            
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        # print(f"before deepsort {time.time()-start}")
        # start =time.time()
        try:
            outputs = deepsort.update(xywhs, confss, oids, im0)
        except:
            print('something wrong')
            pass
        #     return
        # print(f"deepsort {time.time()-start}")
        # start =time.time()
        points = []
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_class = outputs[:, -1]

            for i, box in enumerate(bbox_xyxy):
                points.append([int(box[0]) / im0.shape[1], int(box[1]) / im0.shape[0], int(box[2]) / im0.shape[1], int(box[3]) / im0.shape[0]])
            # assert (np.array(points).copy() < 1).min()

            for index, id_ in enumerate(identities):
                center = ((points[index][0]+points[index][2]) / 2, points[index][3])

                id_ = int(id_)
                if id_ in self.dataframe:
                    self.dataframe[id_]["num_frames"] += self.is_point_in_area(center)
                else:
                    self.dataframe[id_] = {}
                    if object_class[index] == 2:
                        self.dataframe[id_]["label"] = "car"
                    if object_class[index] == 7:
                        self.dataframe[id_]["label"] = "track"
                    if object_class[index] == 5:
                        self.dataframe[id_]["label"] = "buss"
                    self.dataframe[id_]["num_frames"] = self.is_point_in_area(center)
        # print(f"last part {time.time()-start}")
        
        


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
