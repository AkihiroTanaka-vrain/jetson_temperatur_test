import argparse
import os
import glob
import re
import time
import numpy as np
import cv2
from PIL import Image
# import pandas as pd

# pytorch###############
import torch
import torch.nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

#ResNet##################
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152
#########################


def change_padding_mode(network: torch.nn.Module, padding_mode: str = 'reflect') -> None:
    """ 引数として渡されたネットワークのパディングモードを変更する関数
    """
    for module in network.modules():
        if hasattr(module, 'padding_mode'):
            module.padding_mode = padding_mode


def load_model(model, model_path, device, is_train=True):
    if model == 'model1':
        model_t = resnet18(pretrained=False).to(device)
        model_s = resnet18(pretrained=False).to(device)
        model1_path = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'model1.pth')
        model_t.load_state_dict(torch.load(model1_path))
        if not is_train:
            # model_s_path = os.path.join(cfg.OUTPUT_DIR, "vision.pth")
            if device == torch.device('cpu'):
                model_s.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            else:
                model_s.load_state_dict(torch.load(model_path))


    elif model == 'model2':
        model_t = resnet34(pretrained=False).to(device)
        model_s = resnet34(pretrained=False).to(device)
        model2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'model2.pth')
        model_t.load_state_dict(torch.load(model2_path))
        if not is_train:
            # model_s_path = os.path.join(cfg.OUTPUT_DIR, "vision.pth")
            if device == torch.device('cpu'):
                model_s.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            else:
                model_s.load_state_dict(torch.load(model_path))

    elif model == 'model3':
        model_t = resnet50(pretrained=False).to(device)
        model_s = resnet50(pretrained=False).to(device)
        model3_path = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'model3.pth')
        model_t.load_state_dict(torch.load(model3_path))
        if not is_train:
            # model_s_path = os.path.join(cfg.OUTPUT_DIR, "vision.pth")
            if device == torch.device('cpu'):
                model_s.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            else:
                model_s.load_state_dict(torch.load(model_path))
    else:
        raise TypeError("正しいモデル名を入力してください。")

    change_padding_mode(model_t)
    change_padding_mode(model_s)

    return model_t, model_s


# def test(self):
#     print('Test phase start')
#     try:
#         self.model_s.load_state_dict(torch.load(glob.glob(weight_save_path+'/*.pth')[0]))
#         #print(glob.glob(weight_save_path+'/*.pth')[0])
#     except:
#         raise Exception('Check saved model path.')
#     self.model_t.eval()
#     self.model_s.eval()
#     test_path = os.path.join(dataset_path, 'test')
#     #gt_path = os.path.join(dataset_path, 'ground_truth')
#     # 複数拡張子に対応
#     test_imgs = [p for p in glob.glob(test_path + '/*/*', recursive=True) if re.search('/*\.(jpg|jpeg|png|bmp|JPG|PNG)', str(p))]
#     #test_imgs = glob.glob(test_path + '/*/*.bmp', recursive=True)
#     #gt_imgs = glob.glob(gt_path + '/[!good]*/*.png', recursive=True)
#     #gt_val_list = []
#     #pred_val_list = []
#     export_dic = {"file_name":[], "label":[], "AnomalyScore":[]}
#     for i in range(len(test_imgs)):
#         test_img_path = test_imgs[i]
#         #gt_img_path = gt_imgs[i]
#         #assert os.path.split(test_img_path)[1].split('.')[0] == os.path.split(gt_img_path)[1].split('_')[0], "Something wrong with test and ground truth pair!"
#         defect_type = os.path.split(os.path.split(test_img_path)[0])[1]
#         img_name = os.path.split(test_img_path)[1].split('.')[0]

#         # ground truth
#         #gt_img_o = cv2.imread(gt_img_path,0)
#         #gt_img_o = cv2.resize(gt_img_o, (input_size, input_size))
#         #gt_val_list.extend(gt_img_o.ravel()//255)

#         # load image（日本語ファイル名に対応）
#         test_img = Image.open(test_img_path).convert('RGB')
#         start_time = time.time()
#         test_img = test_img.resize((input_size, input_size))
#         test_img_o = np.asarray(test_img)[:,:,::-1]
#         #test_img = np.fromfile(test_img_path, np.uint8)
#         #test_img_o = cv2.imdecode(test_img, cv2.IMREAD_COLOR)
#         #test_img_o = cv2.resize(test_img_o, (input_size, input_size))[:,:,::-1]
#         #test_img = Image.fromarray(test_img_o)
#         test_img = self.test_data_transform(test_img)
#         test_img = torch.unsqueeze(test_img, 0).to(device)
#         with torch.set_grad_enabled(False):
#             self.features_t = extract_resnet_features(test_img, self.model_t)
#             self.features_s = extract_resnet_features(test_img, self.model_s)

#         # ヒートマップの取得
#         anomaly_map, a_maps = cal_anomaly_map(self.features_s, self.features_t, out_size=input_size)
#         #pred_val_list.extend(anomaly_map.ravel())

#         # ヒートマップ色の正規化
#         anomaly_map_norm = min_max_norm(anomaly_map)
#         #anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

#         # 異常度の算出
#         anomaly_score = sum(anomaly_map.ravel())

#         # 64x64 map
#         am64 = min_max_norm(a_maps[0])
#         am64 = cvt2heatmap(am64*255)
#         # 32x32 map
#         am32 = min_max_norm(a_maps[1])
#         am32 = cvt2heatmap(am32*255)
#         # 16x16 map
#         am16 = min_max_norm(a_maps[2])
#         am16 = cvt2heatmap(am16*255)

#         # anomaly map on image
#         if anomaly_score > threshold:
#             heatmap = cvt2heatmap(anomaly_map_norm*255)
#         else:
#             heatmap = cvt2heatmap(anomaly_map_norm*128)
#         hm_on_img = heatmap_on_image(heatmap, test_img_o)
#         # 元画像とヒートマップを横に結合
#         save_img = cv2.hconcat([test_img_o, hm_on_img])
#         h, w, _ = save_img.shape
#         # 異常スコアの表示箇所を生成
#         score_canvas = np.zeros((50, w, 3), np.uint8)
#         # 異常スコア表示箇所と画像を結合
#         save_img = cv2.vconcat([score_canvas, save_img])
#         # 表示する異常スコア
#         anomaly_text = "AnomalyScore : {:.3f}".format(anomaly_score)
#         # 異常スコアを描画
#         save_img = cv2.putText(save_img, anomaly_text, (0, 50//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#         inference_time = time.time() - start_time
#         # save images
#         # 日本語パスに対応
#         #Image.fromarray(np.uint8(test_img_o[:,:,::-1])).save(os.path.join(sample_path, f'{defect_type}_{img_name}.jpg'))
#         #Image.fromarray(np.uint8(hm_on_img[:,:,::-1])).save(os.path.join(sample_path, f'{defect_type}_{img_name}_amap_on_img.jpg'))
#         image_save_path = f'{defect_type}_{img_name}_result.jpg'
#         Image.fromarray(np.uint8(save_img[:,:,::-1])).save(os.path.join(sample_path, image_save_path))
#         export_dic["file_name"].append(img_name)
#         export_dic["label"].append(defect_type)
#         export_dic["AnomalyScore"].append(anomaly_score)
#         #cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}.jpg'), test_img_o)
#         #cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am64.jpg'), am64)
#         #cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am32.jpg'), am32)
#         #cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am16.jpg'), am16)
#         #cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_amap.jpg'), anomaly_map_norm_hm)
#         #cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_amap_on_img.jpg'), hm_on_img)
#         #cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_gt.jpg'), gt_img_o)
#         # 画像名,異常値,推論時間の出力
#         print("-"*20)
#         print("image_name : {}".format(test_img_path.split("\\")[-1]))
#         print("anomaly_score : {:.3f}".format(anomaly_score))
#         print("inference_time : {:.3f} s".format(inference_time))
#     # pd.DataFrame(export_dic).to_csv("./out.csv", index=False)
#     #print('Total test time consumed : {}'.format(time.time() - start_time))
#     #print("Total auc score is :")
#     #print(roc_auc_score(gt_val_list, pred_val_list))
