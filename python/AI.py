# eye predSS_eye.pyを流用

from typing import Any, Dict
import pycocotools
import numpy as np
import torch
import argparse
import json
import os
from collections import defaultdict
import cv2
from skimage import measure
import socket
from concurrent.futures.thread import ThreadPoolExecutor
from time import perf_counter
import time
from torchvision.ops.boxes import batched_nms
import glob

# my modules
import common_utils as cu
from vision.SemSeg.data.defaults import _C as y_cfg
from vision.SemSeg.data import *
from vision.SemSeg.yolact import Yolact
from vision.SemSeg.utils import timer
from vision.SemSeg.utils.augmentations import BaseTransform, FastBaseTransform, Resize
from vision.SemSeg.utils.functions import MovingAverage, ProgressBar
from vision.SemSeg.layers.box_utils import jaccard, center_size, mask_iou
from vision.SemSeg.layers.output_utils import postprocess, undo_image_transformation

import warnings
import copy
import pickle
warnings.simplefilter('ignore')


import csv

def param_to_thr(y_cfg, params):
    class_names = []
    arg_thresholds = {}
    for p in params["labels"]:
        class_names.append(p["name"])
        arg_thresholds[p["name"]] = p["threshold"]
    return class_names, arg_thresholds

class Inferencer:
    def __init__(self, params):

        self.names = []
        self.thresholds = []
        for label in params["labels"]:
            name, threshold = label["name"], label["threshold"]
            self.names.append(name)
            self.thresholds.append(threshold)


        self.use_cuda = True

        self.predictor = Yolact()
        self.predictor.load_weights(params["model_path"])
        self.predictor.requires_grad_(False)
        self.predictor.eval()
        if self.use_cuda:
            self.predictor = self.predictor.cuda()

        self.transform = FastBaseTransform(self.use_cuda, resize=self.predictor.resize_target)

        self.predictor.detect.use_fast_nms = True  # Whether to use a faster, but not entirely correct version of NMS.
        self.predictor.detect.use_cross_class_nms = False  # Whether compute NMS cross-class or per-class.
        cfg.mask_proto_debug = False  # Outputs stuff for scripts/compute_mask.py.
        self.split_result = np.array([])

    def inference(self, img_path):

        img = cv2.imread(img_path)

        frame = torch.from_numpy(img).cuda().float()

        batch = self.transform(frame.unsqueeze(0))

        preds = self.predictor(batch)


import logging

def main():
    count = 0

    img_files = glob.glob(file_path + "/data/image/*jpg")

    inferencer = Inferencer(params)

    print("推論START!!")

    while True:
        # なんとなくの気分で１回毎に推論させる画像ファイルを変更する。
        num = count % len(img_files)
        result = inferencer.inference(img_files[num])
        count += 1
        
if __name__ == '__main__':

    file_path = os.path.dirname(os.path.abspath(__file__) )
    json_path = file_path + "/data/json"
    
    with open(json_path, "r", encoding='utf-8') as jf:
        params = json.load(jf)

    class_names, arg_thresholds = param_to_thr(y_cfg, params)

    y_cfg.DATASET.CLASS_NAMES = class_names
    cfg.replace(make_custom_config(y_cfg, name="eye"))
    cfg.nms_thresh = 0.5

    main()
