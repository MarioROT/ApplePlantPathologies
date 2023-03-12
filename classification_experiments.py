import os
import time
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from numpy import printoptions
import requests
import tarfile
import random
import json
from shutil import copyfile
import pathlib

from typing import List, Dict
from multiprocessing import Pool
from skimage.io import imread
from PIL import Image

import time
import datetime

from tools import get_transform, get_instance_segmentation_model, ObjectDetectionDataSet
from tools import map_class_to_int, save_json, read_json, get_filenames_of_path, adapt_data
from tools import checkpoint_save, show_sample, MutilabelClassificationDataset, evaluate
from tools import MLCDataset
from backbones import GetModel, GetModelNM, BuildModel
from torchmets import sklearn_metrics, compute_full_metrics, compute_tracking_metrics
import shutil
import sys

import re
from math import isnan
from os.path import isfile, join
from neptune.types import File
from torchsummary import summary
from tools import TorchXRayVisionNorm
import itertools as itl

params = {'OWNER': 'rubsini', 
          'PROJECT': 'ApplePlantDiseases',
          'PARTITION':'../Data/Splits/split1.json',
          'IMGS_PATH':'../Data/train_images/', 
          'SAVE_PATH': 'Checkpoints/', 
          'SAVE_DIR':'Result_Tables_Archs', 
          'EXP_TYPE':['Initial Test Exp' ,'General Board'],
          'LOG_MODEL': True,  
          'BACKBONE': 'resnet50', 
          'MODEL_FREEZE': True, 
          'I_WEIGHTS': False,
          'BATCH_SIZE': 16, 
          'LR': 1e-3, 
          'EPOCHS': 50, 
          'N_WORKERS': 1,
          'SAVE_FREQ': 50,
          'WEIGHT_DECAY': 5e-5,
        #   'RESIZE': 224,
          'CROP_SIZE': 512,
          'ROTATION_RANGE': 15,
          'TRANSLATION': 0.10,
          'SCALE': (0.85,1.15),
          "IMG_MEAN":[0.485, 0.456, 0.406],
          "IMG_STD":[0.229, 0.224, 0.225]
          }

import os
import time
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from numpy import printoptions
import requests
import tarfile
import random
import json
from shutil import copyfile
import pathlib

from typing import List, Dict
from multiprocessing import Pool
from skimage.io import imread
from PIL import Image

import time
import datetime

from tools import get_transform, get_instance_segmentation_model, ObjectDetectionDataSet
from tools import map_class_to_int, save_json, read_json, get_filenames_of_path, adapt_data
from tools import checkpoint_save, show_sample, MutilabelClassificationDataset, evaluate
from tools import MLCDataset
from backbones import GetModel, GetModelNM, BuildModel
from torchmets import sklearn_metrics, compute_full_metrics, compute_tracking_metrics
import shutil
import sys

import re
from math import isnan
from os.path import isfile, join
from neptune.new.types import File
from torchsummary import summary
from tools import TorchXRayVisionNorm

params = {'OWNER': 'rubsini',  # Nombre de usuario en Neptune.ai
          'PROJECT': 'Classification-New-Metrics-CDataset', # Nombre dle proyecto creado en Neptune.ai
          'SAVE_PATH': 'Checkpoints/', # Donde Guardar los parametros del modelo entrenado
          'SAVE_DIR':'Result_Tables_CompDS', #'Result_Tables_CompDS',
          'EXP_TYPE':['NF_HIDDEN','Complete Dataset' ,'General Board'],
          'LOG_MODEL': True,  # Si se cargará el modelo a neptune después del entrenamiento
          'BACKBONE': 'densenet121-res224-pc', # Red usada para transferencia de aprendizaje
          'MODEL_FREEZE': True, # Escoger si se congelan los pesos de las capas convolucionales del modelo
          'I_WEIGHTS': False,
          'BATCH_SIZE': 32, # Tamaño del lote
          'LR': 1e-3, # Tasa de aprendizaje
          'EPOCHS': 50, # Número máximo de épocas
          'IOU_THRESHOLD': 0.5, # Umbral de Intersección sobre Union para evaluar predicciones en entrenamiento
          'N_WORKERS': 8,
          'WEIGHT_DECAY': 5e-5,
          'OG_IMAGE_SIZE': '224',
          'PARTITION_USED': 2,
          'DATA_AUGMENTATION_METHOD':1,
          'MODEL_BUILDING_METHOD':2,
          'NF_HIDDEN':True,
          'RESIZE': 224,
          'CROP_SIZE': None,
          'ROTATION_RANGE': 15,
          'TRANSLATION': 10,
          'SCALE': (0.85,1.15) # (tuple) - > (0.85, 1.15)
          }

def main():

    device = torch.device('cuda')

    train_transform = transforms.Compose([
                                        #   transforms.RandomCrop(params['CROP_SIZE']),
                                        # transforms.RandomAffine(params['ROTATION_RANGE'], translate=(params['TRANSLATION'], params['TRANSLATION']), scale=params['SCALE']),
                                        transforms.Resize((params['CROP_SIZE'],params['CROP_SIZE'],)),
                                        # transforms.ToTensor(),
                                        # transforms.Normalize(params["IMG_MEAN"], params["IMG_STD"])
                                        ])
    val_transform = transforms.Compose([
                                        transforms.Resize((params['CROP_SIZE'],params['CROP_SIZE'],)),
                                        # transforms.ToTensor(),
                                        # transforms.Normalize(params["IMG_MEAN"], params["IMG_STD"])
                                        ])

    train_dataset = MLCDataset(params['PARTITION'], params['IMGS_PATH'], 'Train', train_transform)
    val_dataset = MLCDataset(params['PARTITION'], params['IMGS_PATH'], 'Val', train_transform)
    test_dataset = MLCDataset(params['PARTITION'], params['IMGS_PATH'], 'Test', train_transform)

    lt = len(train_dataset)+len(val_dataset)+len(test_dataset)
    ltr,ptr,lvd,pvd,lts,pts = len(train_dataset), len(train_dataset)/lt, len(val_dataset), len(val_dataset)/lt, len(test_dataset), len(test_dataset)/lt
    print('Total data: {} ({:.2f}%)\nTrain data: {} ({:.2f}%)\nVal Data:   {}  ({:.2f}%)\nTest Data:  {}  ({:.2f}%)'.format(lt,lt/lt,ltr,ptr,lvd,pvd,lts,pts))

    train_dataloader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], num_workers=params["N_WORKERS"], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params['BATCH_SIZE'], num_workers=params["N_WORKERS"])
    test_dataloader = DataLoader(test_dataset, batch_size=params['BATCH_SIZE'], num_workers=params["N_WORKERS"])

    test_freq = int(len(train_dataset)/params['BATCH_SIZE'])

    num_train_batches = int(np.ceil(len(train_dataset) / params['BATCH_SIZE']))

    # Initialize the model
    model = GetModelNM(len(train_dataset.classes), params['BACKBONE'], params['MODEL_FREEZE'], params['I_WEIGHTS'])
    # Switch model to the training mode and move it to GPU.
    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'], weight_decay=params['WEIGHT_DECAY'])

    # If more than one GPU is available we can use both to speed up the training.
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    os.makedirs(params['SAVE_PATH'], exist_ok=True)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Logging metadata
    import neptune
    # from neptune.new.types import File  
    NEPTUNE_API_TOKEN = 'Insert KEY Here'
    run = neptune.init_run(project=f'{params["OWNER"]}/{params["PROJECT"]}',
                        api_token=NEPTUNE_API_TOKEN,
                        tags = params['EXP_TYPE'])

    run['parameters'] = params
    # run.stop()

    # Run training
    epoch = 0
    iteration = 0
    best_macro_F1 = 0.0
    while True:
        print('Epoch: ', epoch)
        batch_losses = []
        for imgs, targets in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()

            model_result = model(imgs)
            loss = criterion(model_result, targets.type(torch.float))

            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            run["logs/lr"].log(optimizer.param_groups[0]['lr'])
            run["logs/loss_step"].log(loss)

            batch_losses.append(batch_loss_value)
            with torch.no_grad():
                uap = compute_tracking_metrics(targets.cpu().numpy(), torch.sigmoid(model_result).cpu().numpy())
                run["logs/train_uap"].log(uap)
                result = sklearn_metrics(torch.sigmoid(model_result).cpu().numpy(), targets.cpu().numpy())
                # for metric in result:
                    #  run["logs/train_"+metric.replace('/','_')].log(result[metric])

            if iteration % test_freq == 0:
                model.eval()
                with torch.no_grad():
                    model_result = []
                    targets = []
                    for imgs, batch_targets in val_dataloader:
                        imgs = imgs.to(device)
                        model_batch_result = torch.sigmoid(model(imgs))
                        model_result.extend(model_batch_result.cpu().numpy())
                        targets.extend(batch_targets.cpu().numpy())

                result = sklearn_metrics(np.array(model_result), np.array(targets))
                # res_macro_F1 = result['macro/f1']
                uap = compute_tracking_metrics(np.array(targets), np.array(model_result))
                run["logs/validation_uap"].log(uap)
                res_macro_F1 =  uap
                #for metric in result:
                    # logger.add_scalar('test/' + metric, result[metric], iteration)
                #    run["logs/test_"+metric.replace('/','_')].log(result[metric])
                print("epoch:{:2d} iter:{:3d} TEST: "
                        "micro f1: {:.3f} "
                        "macro f1: {:.3f} "
                        "samples f1: {:.3f} "
                        "uap: {:.3f}".format(epoch, iteration,
                                                    result['micro/f1'],
                                                    result['macro/f1'],
                                                    result['samples/f1'],
                                                    uap))

                model.train()
            iteration += 1

        loss_value = np.mean(batch_losses)
        run["logs/loss_epoch"].log(loss_value)
        print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
        print("epoch:{:2d} iter:{:3d} TRAIN: "
                    "micro f1: {:.3f} "
                    "macro f1: {:.3f} "
                    "samples f1: {:.3f} "
                    "uap: {:.3f}".format(epoch, iteration,
                                                result['micro/f1'],
                                                result['macro/f1'],
                                                result['samples/f1'],
                                                uap))
        if best_macro_F1 < res_macro_F1 or epoch % save_freq == 0:
            if best_macro_F1 <= res_macro_F1:
                best_macro_F1 = res_macro_F1
                s_model, s_epoch = model, epoch
        if epoch % save_freq == 0:
            checkpoint_save(s_model, save_path, s_epoch, run)
        epoch += 1
        if max_epoch_number < epoch:
            break

    #params["EXP_TYPE"][0] = params["EXP_TYPE"][1]+"/"+params["EXP_TYPE"][0]
    os.makedirs(params["SAVE_DIR"]+"/"+params["EXP_TYPE"][0], exist_ok=True)
    params["RUN"] = run
    params["PAT_NAMES"] = [i.capitalize() for i in list(train_dataset.classes)] # list(mapping.keys())
    print('PARAMS --> PAT_NAMES', params['PAT_NAMES'])
    res = [i.start() for i in re.finditer('/', run.get_url())]
    params['EXP_NAME'] = run.get_url()[max(res)+1:]
    best_epoch = max(get_filenames_of_path(pathlib.Path('Checkpoints')))
    model.load_state_dict(torch.load(best_epoch))
    params['BEST_EPOCH'] = [int(s) for s in best_epoch.name.split('-')[1].split('.') if s.isdigit()][0]
    evaluate(model, val_dataloader, test_dataloader, params)

    run.stop()
    shutil.rmtree("Checkpoints")



if __name__ == "__main__":
    main()
