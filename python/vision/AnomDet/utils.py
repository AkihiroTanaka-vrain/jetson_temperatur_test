import argparse
import os
import glob
import re
import time
from typing import Sequence, Union
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# pytorch###############
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def cal_loss(fs_dict, ft_dict):
    tot_loss = 0
    for layer in list(fs_dict.keys()):
        fs = fs_dict[layer]
        ft = ft_dict[layer]
        n, c, h, w = fs.shape
        # 論文通りに計算を実施
        ## それぞれのノルムを正規化
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        #fs_norm = torch.divide(fs, torch.norm(fs, p=2, dim=1, keepdim=True))
        #ft_norm = torch.divide(ft, torch.norm(ft, p=2, dim=1, keepdim=True))
        a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        f_loss = (1/(w*h))*torch.sum(a_map)

        ## 画像サイズで割ってスケール統一
        #f_loss = 0.5*criterion(fs_norm, ft_norm)/(w*h)

        tot_loss += f_loss
    return tot_loss

def cal_anomaly_map(fs_dict, ft_dict, out_size: Union[int, Sequence[int]] = 256):
    # ペアワイズ距離の計算方法を指定
    pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
    # ベースとなるマップの作成
    if isinstance(out_size, int):
        anomaly_map = np.zeros([out_size, out_size])
    else:
        anomaly_map = np.zeros(out_size)
    # 各レイヤーブロックごとの特徴抽出結果のマップを保存するリスト
    a_map_list = []
    for layer in list(fs_dict.keys()):
        fs = fs_dict[layer]
        ft = ft_dict[layer]

        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        a_map = torch.unsqueeze(a_map, dim=1)

        # 出力サイズに合わせるために補完する
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
        a_map = a_map[0,0,:,:].to('cpu').detach().numpy()
        # 中間層ごとにマップを保存
        a_map_list.append(a_map)
        # ベースのマップにかけ合わせる
        anomaly_map += a_map
    n_layers = len(fs_dict.keys())
    return anomaly_map / n_layers, a_map_list

def extract_resnet_features(inputs, model):
    features = dict()
    # extract stem features as level 1
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    features["layer1"] = x
    x = model.layer2(x)
    features["layer2"] = x
    x = model.layer3(x)
    features["layer3"] = x

    return features

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)
