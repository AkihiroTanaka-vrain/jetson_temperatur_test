import os
# from vision.ObjDet.dataset.transform import image_transform
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# import vision.common_utils as cu
from PIL import Image
import cv2
from pycocotools.coco import COCO

# ImageNet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


def format_dataset(dataset_path):
    path_list, label_list =  [], []

    with open(dataset_path, "r") as f:
        for data in f.readlines():
            path, label = data.split(",")
            label = int(label)
            path_list.append(path), label_list.append(label)

    return path_list, label_list

class build_dataset(data.Dataset):
    def __init__(self, cfg, is_train):
        # if is_train:
        #     #self.coco = COCO(cfg.DATASETS.TRAIN)
        #     self.coco = COCO(cfg['DATASET']['TRAIN_JSON'])
        # else:
        #     self.coco = COCO(cfg['DATASET']['VAL_JSON'])

        # #self.transform = build_transforms(cfg.INPUT, is_train)
        # self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        # categories = self.coco.loadCats(self.coco.getCatIds())
        # categories.sort(key=lambda x: x['id'])

        self.classes = {}
        # self.coco_labels = {}
        # self.coco_labels_inverse = {}
        # for c in categories:
        #     self.coco_labels[len(self.classes)] = c['id']
        #     self.coco_labels_inverse[c['id']] = len(self.classes)
        #     self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
    def __len__(self):
        return len(self.image_ids)

    # def __getitem__(self, index):
    #     path = self.load_path(index)
    #     img = cu.imread(path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     #img = Image.open(path).convert('RGB')
    #     img  = Image.fromarray(img)
    #     #img_transformed = self.transform(img)
    #     transformes = self.train_data_transforms()
    #     img_transformed = transformes(img)
    #     label = self.load_cat(index)

    #     return img_transformed, label

    def load_path(self,image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        if not "path" in image_info.keys() or not image_info["path"]:
            images_folder = os.path.join(os.path.dirname(self.coco_json), "data")
            path = os.path.join(images_folder, image_info["file_name"])
        else:
            path = image_info["path"]
        return path
    def load_cat(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index])
        coco_annotations = self.coco.loadAnns(annotations_ids)
        label = int(coco_annotations[0]['category_id'])
        return label

    def train_data_transforms(self, input_size = 224, mean_train=mean_train, std_train=std_train):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
        return data_transforms

    def test_data_transforms(self, input_size = 224, mean_train=mean_train, std_train=std_train):
        data_transforms = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train,
                                    std=std_train)])
        return data_transforms


def make_dataset(cfg, is_train=True):
    dataset = build_dataset(cfg, is_train)
    if is_train:
        data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=True)
    else:
        data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
        #data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=False)

    return data_loader

"""
def load_dataset(dataset_path, batch_size):
    image_datasets = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_data_transforms)
    dataloaders = DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    dataset_sizes = {'train': len(image_datasets)}
    print('Train Dataset Size {}'.format(dataset_sizes['train']))
"""
