from typing import List, Dict
import pathlib
from multiprocessing import Pool
from skimage.io import imread
from skimage.color import rgb2gray,rgba2rgb

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image

import time
import datetime
import json

import utils
# import transforms as T
from matplotlib import pyplot as plt
from numpy import printoptions
import itertools as itl

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch import nn

# import torchxrayvision as xrv
from tqdm import tqdm
import re
from math import isnan
from os.path import isfile, join
from neptune.new.types import File

from torchmets import compute_full_metrics


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """
    Regresa una lista de archivos en un directorio, dado como objeto de pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames

def read_json(path: pathlib.Path):
    with open(str(path), "r") as fp:
        file = json.loads(s=fp.read())
        fp.close()
    return file

def save_json(obj, path: pathlib.Path):
    with open(path, "w") as fp:
        json.dump(obj=obj, fp=fp, indent=4, sort_keys=False)

def map_class_to_int(labels: List[str], mapping: dict):
    """Mapea una cadena (string) a un entero (int)."""
    labels = np.array(labels)
    dummy = np.empty_like(labels)
    for key, value in mapping.items():
        dummy[labels == key] = value

    return dummy.astype(np.uint8)

class ObjectDetectionDataSet(torch.utils.data.Dataset):
    """
    Construye un conjunto de datos con imágenes y sus respectivas etiquetas (objetivos).
    Cada target es esperado que se encuentre en un archivo JSON individual y debe contener
    al menos las llaves 'boxes' y 'labels'.
    Las entradas (imágenes) y objetivos (etiquetas) son esperadas como una lista de
    objetos pathlib.Path

    En caso de que las etiquetas esten en formato string, puedes usar un diccionario de
    mapeo para codificarlas como enteros (int).

    Regresa un diccionario con las siguientes llaves: 'x', 'y'->('boxes','labels'), 'x_name', 'y_name'
    """

    def __init__(
        self,
        inputs: List[pathlib.Path],
        targets: List[pathlib.Path],
        transform = None,
        add_dim: bool = False,
        use_cache: bool = False,
        convert_to_format: str = None,
        mapping: Dict = None,
        tgt_int64: bool = False,
        metadata_dir: pathlib.Path = None,
        filters: List = None,
        id_column: str = None
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.add_dim = add_dim
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping
        self.tgt_int64 = tgt_int64
        self.metadata = metadata_dir
        self.filters = filters
        self.id_column = id_column

        if self.use_cache:
            # Usar multiprocesamiento para cargar las imagenes y las etiquetas en la memoria RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets))

        if metadata_dir:
            self.filtered_inputs = []
            self.filtered_targets = []
            self.id_list = self.add_filters(self.metadata, self.filters, self.id_column)
            for num,input in enumerate(self.inputs):
                if re.search(r'.*\\(.*)\..*', str(input)).group(1) in self.id_list:
                    self.filtered_inputs.append(input)
                    self.filtered_targets.append(self.targets[num])
            self.inputs = self.filtered_inputs
            self.targets = self.filtered_targets


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Seleccionar una muestra
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Cargar entradas (imágenes) y objetivos (etiquetas)
            x, y = self.read_images(input_ID, target_ID)

        # # De RGBA a RGB
        # if x.shape[-1] == 4:
        #     x = rgba2rgb(x)

        # Leer cajas
        try:
            boxes = torch.from_numpy(y["boxes"]).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y["boxes"]).to(torch.float32)

        # Leer puntajes
        if "scores" in y.keys():
            try:
                scores = torch.from_numpy(y["scores"]).to(torch.float32)
            except TypeError:
                scores = torch.tensor(y["scores"]).to(torch.float32)

        # Mapeo de etiquetas
        if self.mapping:
            labels = map_class_to_int(y["labels"], mapping=self.mapping)
        else:
            labels = y["labels"]

        # Leer etiquetas
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Convertir formato
        if self.convert_to_format == "xyxy":
            boxes = box_convert(
                boxes, in_fmt="xywh", out_fmt="xyxy"
            )  # Transformaciones de las cajas del formato xywh a xyxy
        elif self.convert_to_format == "xywh":
            boxes = box_convert(
                boxes, in_fmt="xyxy", out_fmt="xywh"
            )  # # Transformaciones de las cajas del formato xyxy a xywh

        # Crear objetivos
        tgt = {"boxes": boxes, "labels": labels}

        if "scores" in y.keys():
            target["scores"] = scores

        # Preprocesamiento
        tgt = {
            key: value.numpy() for key, value in tgt.items()
        }  # Todos los tensores debieren ser convertidos a np.ndarrays

        if self.transform is not None:
            x, tgt = self.transform(x, tgt)  # Regresa np.ndarrays

        if "scores" in y.keys():
            bxs,lbs,srs = [],[],[]
            for r,f in enumerate(tgt['scores']):
                if f > 0.70:
                    bxs.append(tgt['boxes'][r])
                    lbs.append(tgt['labels'][r])
                    srs.append(tgt['scores'][r])
            tgt = {'boxes':np.array(bxs), 'labels':np.array(lbs), 'scores':np.array(srs)}

        # Agregar Dimensión
        if self.add_dim == 3:
            if len(x.shape) == 2:
                # x = x.T
                # x = np.array([x])
                xD = np.empty((3,x.shape[0],x.shape[1]))
                xD[0],xD[1],xD[2] = x,x,x
                # xD =  np.moveaxis(xD,source=0, destination=-1)
                x = xD
            elif len(x.shape) == 3:
                # f = 1
                x = np.moveaxis(x,source=-1, destination=0)
        elif self.add_dim == 2:
            if len(x.shape) == 2:
                x = x.T
                x = np.array([x])
            elif len(x.shape) == 3:
                x = np.moveaxis(x,source=-1, destination=0)
                x = x[0].T
                x = np.array([x])
            # print(x.shape)
            # x = np.moveaxis(x, source=1, destination=-1)
            # x = np.expand_dims(x, axis=0)

        # print('Before: ', target)
        # Encasillar
        if self.tgt_int64:
            x = torch.from_numpy(x).type(torch.float32)
            tgt = {
                key: torch.from_numpy(value).type(torch.int64)
                for key, value in tgt.items()
            }
        else:
            x = torch.from_numpy(x).type(torch.float32)
            tgt = {
                key: torch.from_numpy(value).type(torch.float64)#int64)
                for key, value in tgt.items()
            }
        # print('After: ', target)

        boxes = tgt['boxes']
        labels = tgt['labels']

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = torch.from_numpy(boxes)
        target["labels"] = torch.from_numpy(labels)
        target["image_id"] = image_id
        target["area"] = torch.from_numpy(area)
        target["iscrowd"] = iscrowd
        target["x_name"] = torch.tensor(int(self.inputs[index].name[:-4].replace('_','')))

        return x, target

    @staticmethod
    def read_images(inp, tar):
        return Image.open(inp).convert("RGB"), read_json(tar) #read_pt(tar)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                      #  hidden_layer,
                                                      #  num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class MutilabelClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, anno_path, transforms, mod_dims = False):
        self.transforms = transforms
        self.mod_dims = mod_dims
        with open(anno_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        self.classes = json_data['labels']

        self.imgs = []
        self.annos = []
        self.data_path = data_path
        print('loading', anno_path)
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            self.annos[item_id] = np.array(vector, dtype=float)
            # print("Item ID: ", item_id,'Item: ', item, 'Vector:', vector, 'Annos: ', self.annos)

    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.data_path, self.imgs[item])
        img = Image.open(img_path)
        if self.mod_dims and (img.mode == 'L' or img.mode == 'RGBA'):
          img = img.convert('RGB')
        elif self.mod_dims and (img.mode == 'RGB' or img.mode == 'RGBA'):
          img = img.convert('L')
        print(img.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)

def adapt_data(data, classes, file_name):
    inputs, targets = data
    samples = []
    for i in range(len(targets)):
        samples.append({"image_name":inputs[i].__str__()[-16:],"image_labels":read_json(targets[i])['labels']})
    Js = {"samples":samples, "labels": list(classes.keys())}
    save_json(Js, pathlib.Path(file_name))

def checkpoint_save(model, save_path, epoch, run):
    f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    run["Model States/checkpoint-{:06d}.pth".format(epoch)].upload(f)
    print('saved & uploaded checkpoint:', f)

def show_sample(img, binary_img_labels):
  # Convert the binary labels back to the text representation.
  img_labels = np.array(dataset_val.classes)[np.argwhere(binary_img_labels > 0)[:, 0]]
  plt.imshow(img)
  plt.title("{}".format(', '.join(img_labels)))
  plt.axis('off')
  plt.show()

def labels2tensornums(targets, mappingR):
  num_labels = []
  for labs in targets:
    n_lab = []
    for lab in labs: 
      n_lab.append(mappingR[lab])
    num_labels.append(torch.tensor(n_lab))
  return num_labels

# Simple dataloader and label binarization, that is converting test labels into binary arrays number of classes) with 1 in places of applicable labels).
class MutilabelClassificationEntireDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, annos, transforms, tdim = None):
        self.transforms = transforms
        self.tdim = tdim
        self. Metadata = pd.read_csv('Complete/Data_Entry_2017_v2020.csv')
        self.classes = sorted(set([lab for labs in self.Metadata["Finding Labels"].unique() for lab in labs.split("|")]))

        self.imgs = imgs
        self.annos = annos
        self.labels = {}
        for item_id in self.imgs:
            items = self.annos[item_id.name]
            vector = [cls in items for cls in self.classes]
            self.labels[item_id.name] = np.array(vector, dtype=float)
            # print("Item ID: ", item_id.name,'Item: ', items, 'Vector:', vector, 'Annos: ', self.annos)

    def __getitem__(self, item):
        img_path = os.path.join(self.imgs[item])
        img = Image.open(img_path)
        if self.tdim:
          img = img.convert("RGB")
        anno = self.labels[self.imgs[item].name]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)

# Simple dataloader and label binarization, that is converting test labels into binary arrays of length 27 (number of classes) with 1 in places of applicable labels).
class MutilabelClassificationTorchXrayEntireDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, annos, transforms, extended_labels = False):
        self.transforms = transforms
        self.extended_labels = extended_labels
        self.Metadata = pd.read_csv('Complete/Data_Entry_2017_v2020.csv')
        self.classes = sorted(set([lab for labs in self.Metadata["Finding Labels"].unique() for lab in labs.split("|")]))

        self.imgs = imgs
        self.annos = annos
        self.labels = {}
        for item_id in self.imgs:
            items = self.annos[item_id.name]
            vector = [cls in items for cls in self.classes]
            if self.extended_labels:
              vector = vector + [0 for i in range(self.extended_labels)]
            self.labels[item_id.name] = np.array(vector, dtype=float)
            # print("Item ID: ", item_id.name,'Item: ', items, 'Vector:', vector, 'Annos: ', self.annos)

    def __getitem__(self, item):
        img_path = os.path.join(self.imgs[item])
        img = imread(img_path)
        # img = xrv.datasets.normalize(img, maxval=255, reshape=True)
        anno = self.labels[self.imgs[item].name]
        if self.transforms is not None:
            img = xrv.datasets.normalize(img, maxval=255, reshape=True)
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)

# Simple dataloader and label binarization, that is converting test labels into binary arrays of length 27 (number of classes) with 1 in places of applicable labels).
class MutilabelClassificationTorchXrayDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, anno_path, transforms, extended_labels = False):
        self.transforms = transforms
        self.extended_labels = extended_labels
        with open(anno_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        self.classes = json_data['labels']

        self.imgs = []
        self.annos = []
        self.data_path = data_path
        print('loading', anno_path)
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            if self.extended_labels:
              vector = vector + [0 for i in range(self.extended_labels)]
            self.annos[item_id] = np.array(vector, dtype=float)
            # print("Item ID: ", item_id,'Item: ', item, 'Vector:', vector, 'Annos: ', self.annos)

    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.data_path, self.imgs[item])
        img = imread(img_path)
        if self.transforms is not None:
            img = xrv.datasets.normalize(img, maxval=255, reshape=True)
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)

# --------------------- Datasets for new metrics experiments ------------------- #
class TorchXRayVisionNorm(nn.Module):
    """Normalizes numpy image from [0, maxval] to [-1024, 1024].
    See:
        https://github.com/mlmed/torchxrayvision/issues/9
        https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def forward(self, x):
        if x.dtype == torch.float32:
            return (2 * x / 255 - 1) * 1024
        if x.dtype == torch.uint8:
            return (2 * (x.type(torch.float32) / 255) - 1) * 1024
        if x.dtype == torch.uint16:
            return (2 * (x.type(torch.float32) / 65535) - 1) * 1024
        raise ValueError(f'unknown dtype={x.dtype}')

# class MLCDataset(torch.utils.data.Dataset):
#     def __init__(self, csv_path, imgs_path, partition, transforms = None, mod_dims = False, xrv_norm=False, noFinding_hidden=False):
#         self.transforms = transforms
#         self.mod_dims = mod_dims
#         self.xrv_norm = xrv_norm
#         self.noFinding_hidden = noFinding_hidden
#         partitions = {'train':0,'validation':1,'test':2}
#         self.csv_path = csv_path
#         self.imgs_path = imgs_path
#         self.csv = pd.read_csv(self.csv_path)
#         self.csv = self.csv[self.csv.subset == partitions[partition]]
#         if not noFinding_hidden:
#             self.classes = self.csv.columns[1:-1].values
#         else:
#             self.classes = [clas for clas in self.csv.columns[1:-1].values if clas != 'no finding']

#         self.img_names = self.csv["example_id"].values
#         if not self.noFinding_hidden:
#           self.labels = [self.csv.iloc[i][1:-1].values for i in range(len(self.csv))]
#         else:
#           self.labels = [np.delete(self.csv.iloc[i][1:-1].values,10) for i in range(len(self.csv))]
#         print('loading', self.csv_path)

#     def __getitem__(self, item):
#         label = torch.tensor(self.labels[item].astype(np.int32))
#         img_path = os.path.join(self.imgs_path, self.img_names[item]+'.png')
#         img = Image.open(img_path)
#         if self.mod_dims and (img.mode == 'L' or img.mode == 'RGBA' or img.mode == 'LA'):
#           img = img.convert('RGB')
#         elif self.mod_dims and (img.mode == 'RGB' or img.mode == 'RGBA' or img.mode == 'LA'):
#           img = img.convert('L')
#         if self.xrv_norm:
#             try:
#                 img = rgb2gray(imread(img_path))
#             except:
#                 try:
#                     img = rgb2gray(rgba2rgb(imread(img_path)))
#                 except:
#                     img = imread(img_path)
#             img = xrv.datasets.normalize(img, maxval=img.max(), reshape=True)
#             img = torch.from_numpy(img).type(torch.float32)
#             #print(img.min(), img.max())
#             #img = torch.unsqueeze(torch.from_numpy(img),0).type(torch.float32)
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img, label

#     def __len__(self):
#         return len(self.img_names)

class MLCDataset(torch.utils.data.Dataset):
    def __init__(self,  json_path, imgs_path, partition, transforms = None):
        self.transforms = transforms
        self.json_path = json_path
        self.imgs_path = imgs_path
        self.json = read_json(pathlib.Path(json_path))[partition]
        self.classes = sorted(list(set(itl.chain(*self.json.values()))))
        self.class_to_index = {c:n for n,c in enumerate(self.classes)}
        self.img_names = list(self.json.keys())
        self.labels = self.encode(list(self.json.values()), self.class_to_index)

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item].astype(np.int32))
        img_path = os.path.join(self.imgs_path, self.img_names[item])
        img = imread(img_path)
        # img = torch.Tensor(imread(img_path))
        # img = torch.moveaxis(img, -1, 0)
        # img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.img_names)

    @staticmethod
    def encode(y, mapping):
        '''Numeric to OneHot'''
        y_one_hot = np.zeros((len(y), len(mapping)), dtype=int)
        for i, item in enumerate(y):
            for class_ in item:
                y_one_hot[i][mapping[class_]] = 1
        return y_one_hot
    
class MLCDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.lenD = len(dataset)
    def __getitem__(self,item):
        mn = item * self.batch_size
        mx = (item * self.batch_size) + self.batch_size
        imgs = []
        labels = []
        if mx < len(self.dataset):
            for i in range(mn, mx):
                imgs.append(self.dataset[i][0][None,:,:,:])
                labels.append(self.dataset[i][1][None,:])
        else:
            for i in range(mn, len(self.dataset)):
                imgs.append(self.dataset[i][0][None,:,:,:])
                labels.append(self.dataset[i][1][None,:])
        imgs = torch.cat(imgs, 0).to('cuda')
        labels = torch.cat(labels, 0).to('cuda')
        return imgs, labels
    def __len__(self):
        return int(self.lenD / self.batch_size) + 1

class HNFDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, imgs_path, partition, transforms = None, mod_dims = False, xrv_norm=False, noFinding_hidden=False, nf_percentage = None):
        self.transforms = transforms
        self.mod_dims = mod_dims
        self.xrv_norm = xrv_norm
        self.noFinding_hidden = noFinding_hidden
        self.nf_percentage = nf_percentage
        partitions = {'train':0,'validation':1,'test':2}
        self.csv_path = csv_path
        self.imgs_path = imgs_path
        self.csv = pd.read_csv(self.csv_path)
        self.csv = self.csv[self.csv.subset == partitions[partition]]

        if not self.noFinding_hidden and not self.nf_percentage:
            self.classes = self.csv.columns[1:-1].values
        else:
            self.classes = [clas for clas in self.csv.columns[1:-1].values if clas != 'no finding']

        if not self.noFinding_hidden and not self.nf_percentage:
          self.labels = [self.csv.iloc[i][1:-1].values for i in range(len(self.csv))]
          self.img_names = self.csv["example_id"].values
        else:
          self.labels =  self.csv[self.csv["no finding"] == 0]
          self.labels = [np.delete(self.labels.iloc[i][1:-1].values,10) for i in range(len(self.labels))]
          self.img_names = self.csv[self.csv["no finding"] == 0]["example_id"].values
          if self.nf_percentage:
              N_samples =  round(len(self.csv)*self.nf_percentage)
              NFs = self.csv[self.csv["no finding"] == 1].sample(N_samples)
              NF_labels = [np.delete(NFs.iloc[i][1:-1].values,10) for i in range(len(NFs))]
              NF_images = NFs["example_id"].values
              self.labels += NF_labels
              self.img_names = np.append(self.img_names, NF_images)

        print('loading', self.csv_path)

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item].astype(np.int32))
        img_path = os.path.join(self.imgs_path, self.img_names[item]+'.png')
        img = Image.open(img_path)
        if self.mod_dims and (img.mode == 'L' or img.mode == 'RGBA' or img.mode == 'LA'):
          img = img.convert('RGB')
        elif self.mod_dims and (img.mode == 'RGB' or img.mode == 'RGBA' or img.mode == 'LA'):
          img = img.convert('L')
        if self.xrv_norm:
            try:
                img = rgb2gray(imread(img_path))
            except:
                try:
                    img = rgb2gray(rgba2rgb(imread(img_path)))
                except:
                    img = imread(img_path)
            img = xrv.datasets.normalize(img, maxval=img.max(), reshape=True)
            img = torch.from_numpy(img).type(torch.float32)
            #print(img.min(), img.max())
            #img = torch.unsqueeze(torch.from_numpy(img),0).type(torch.float32)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.img_names)

def predict(model, dl, subset):
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        print(dl)
        for batch in tqdm(dl, ncols=75, desc=f'Eval {subset}'):
            x = batch[0].to(device)
            y_true = batch[1].to(device)
            y_prob = torch.sigmoid(model(x))
            outputs.append([y_true, y_prob])
        y_true, y_prob = list(zip(*outputs))
        y_true = torch.cat(y_true)
        y_prob = torch.cat(y_prob)
    y_prob = y_prob.cpu().numpy()
    y_true = y_true.cpu().numpy()
    return y_true, y_prob


def save_results(metrics, params, subset,):
    cols = ['run','epoch', 'uap', 'map', 'wap', 'iap', 'f1', 'prec', 'rec'] + params["PAT_NAMES"]
    metrics = [m * 100 for m in metrics]
    df = pd.DataFrame(columns=cols)
    df.loc[0] = [params['EXP_NAME'], params['BEST_EPOCH']] + metrics

    name = f'{subset}_csv'
    path = join(params["SAVE_DIR"], params["EXP_TYPE"][0], f'{name}.csv')
    if isfile(path):
        df = pd.concat([pd.read_csv(path, index_col=None), df])
    df.to_csv(path, index=False, float_format='%.2f')

    params["RUN"][f'Artifacts/{name}'].upload(File.as_html(df))
    # log_table(name=f'{name}', table=df, experiment=params["RUN"])


def evaluate(model, val_dl, tst_dl, params):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    model.to(device)
    model.eval()
    for dl, subset in [[val_dl, 'val'], [tst_dl, 'tst']]:
        y_true, y_prob = predict(model, dl, subset)
        metrics = compute_full_metrics(y_true, y_prob)
        save_results(metrics, params, subset)
