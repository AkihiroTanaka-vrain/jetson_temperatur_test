import cv2
import numpy as np
import torch
from torch.autograd import Function
import json
import time
import torchvision
import torchvision.transforms.functional as F
import os
from torch.nn import Softmax
from vision.classification.data.transform import build_transforms
from PIL import Image, ImageDraw, ImageFont
from common_utils.custom_cv2 import calc_box_size


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        bs = x.size(0)
        x = self.model.extract_features(x)
        x.register_hook(self.save_gradient)
        outputs += [x]
        x = self.model._avg_pooling(x)

        return outputs, x


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._fc(output)
        return target_activations, output


class GradCam:
    def __init__(self, json_params, model, target_layer_names, use_cuda, out_features):
        self.json_params = json_params
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.resize, self.size = json_params['input']['resize_use'], json_params['input']['resize_size']
        if self.cuda and torch.cuda.get_device_properties(0).total_memory > 7864320000:
            self.device =  "cuda:0"
        else:
            self.device =  "cpu"
        self.model = model.to(self.device)

        self.extractor = ModelOutputs(self.model, target_layer_names)
        self.labels = json_params['class_names']
        self.colors = np.array(
        [
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            1.000, 1.000, 0.000,
            1.000, 0.333, 0.500,
            0.000, 0.333, 1.000,
            0.000, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.000, 0.000,

        ]
    ).astype(np.float32).reshape(-1, 3)

    def forward(self, input):
        return self.model(input)

    def cam2(self, img, img_tensor, prob, width, height):
        mask, index, prob = self.__call__(img_tensor)
        img = np.array(img, dtype=np.float32) / 255
        mask = cv2.resize(mask, (width, height))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_img = heatmap + np.float32(img)
        cam_img = (cam_img / np.max(cam_img)) * 255

        label = self.labels[index]
        score = prob[index]
        text_str = "{}: {}%".format(label.replace("\n", ""), int(score*100))
        text_color = tuple([int(c) for c in self.colors[index % self.colors.shape[0]] * 255])
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thickness, box_size = calc_box_size(cam_img, bias=1)

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(cam_img, (0, 0), (width-box_size, height-box_size), text_color, box_size)
        cv2.rectangle(cam_img, (0, 0), (text_w+3, text_h+3), text_color, -1)
        cv2.putText(cam_img, text_str, (0,text_h), font_face, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
        cam_img = cv2.addWeighted(cam_img, 0.7, img, 1 - 0.7, 0)
        return cam_img

    def cam(self, img, prob, image_label_list=None, plot_gradcam=True, is_set_label=True):
        H, W, _ = img.shape
        image_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(image_array)
        if self.resize:
            img_tensor = F.resize(img_tensor, self.size, Image.BILINEAR)
        img_tensor = F.normalize(img_tensor, self.json_params['input']['normalize_mean'], self.json_params['input']['normalize_std'], False).unsqueeze(0)
        if plot_gradcam:
            mask, index, prob = self.__call__(img_tensor)
            img = np.array(img, dtype=np.float32) / 255
            mask = cv2.resize(mask, (W, H))
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam_img = heatmap + np.float32(img)
            cam_img = (cam_img / np.max(cam_img)) * 255
        else:
            with torch.no_grad():
                output = self.model(img_tensor.to(self.device))
                prob = Softmax(dim=1)(output)[0].tolist()
                index = np.argmax(output.cpu().data.numpy())
                cam_img = img

        if is_set_label:
            label = self.labels[index]
            score = prob[index]
            text_str = "{}: {}%".format(label.replace("\n", ""), int(score*100))
            text_color = tuple([int(c) for c in self.colors[index % self.colors.shape[0]] * 255])
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, font_thickness, box_size = calc_box_size(cam_img, bias=1)

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(cam_img, (0, 0), (W-box_size, H-box_size), text_color, box_size)
            cv2.rectangle(cam_img, (0, 0), (text_w+3, text_h+3), text_color, -1)
            cv2.putText(cam_img, text_str, (0,text_h), font_face, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
            cam_img = cv2.addWeighted(cam_img, 0.7, img, 1 - 0.7, 0)
        return prob, cam_img

    def __call__(self, input, index=None):
        _, _, H, W = input.size()
        if self.cuda and torch.cuda.get_device_properties(0).total_memory > 7864320000:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        prob = Softmax(dim=1)(output)[0].tolist()
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (W, H))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam, index, prob


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output
