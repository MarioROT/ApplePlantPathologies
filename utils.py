import json
import os
import pathlib
import itertools

import importlib_metadata
import numpy as np
import pandas as pd
import torch
from IPython import get_ipython
from skmultilearn.model_selection import IterativeStratification
# from neptunecontrib.api import log_table
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
# from torchvision.ops import box_convert, box_area
# import torchvision.transforms as TIM

# from metrics.bounding_box import BoundingBox
# from metrics.enumerators import BBFormat, BBType

from scipy.signal import savgol_filter


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """
    Regresa una lista de archivos en un directorio, dado como objeto de pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames

def select_interpolation_method(method):
    imc = {'bilinear':TIM.InterpolationMode.BILINEAR,
           'nearest':TIM.InterpolationMode.NEAREST,
           'bicubic':TIM.InterpolationMode.BICUBIC,
           'box':TIM.InterpolationMode.BOX,
           'hamming':TIM.InterpolationMode.HAMMING,
           'lanczos':TIM.InterpolationMode.LANCZOS}
    return imc[method]

def read_pt(path: pathlib.Path):
    file = torch.load(path)
    return file

def read_json(path: pathlib.Path):
    with open(str(path), "r") as fp:
        file = json.loads(s=fp.read())
        fp.close()
    return file


def save_json(obj, path: pathlib.Path):
    with open(path, "w") as fp:
        json.dump(obj=obj, fp=fp, indent=4, sort_keys=False)


def collate_double(batch):
    """
    Función usada por el cargador de datos para cotejar el objeto ObjectDetectionDataSet.
    """
    x = [sample["x"] for sample in batch]
    y = [sample["y"] for sample in batch]
    x_name = [sample["x_name"] for sample in batch]
    y_name = [sample["y_name"] for sample in batch]
    return x, y, x_name, y_name


def collate_single(batch):
    """
    Función usada por el cargador de datos para cotejar el objeto ObjectDetectionDataSetSingle.
    """
    x = [sample["x"] for sample in batch]
    x_name = [sample["x_name"] for sample in batch]
    return x, x_name


def color_mapping_func(labels, mapping):
    """Mapea etiquetas en formato entero o cadena a un color cada una."""
    color_list = [mapping[value] for value in labels]
    return color_list


def enable_gui_qt():
    """Desempeña el comando mágico %gui qt"""
    ipython = get_ipython()
    ipython.magic("gui qt")


# def stats_dataset(dataset, rcnn_transform: GeneralizedRCNNTransform = False):
#     """
#     Itera sobre el conjunto de datos y regresa algunas estadísticas del mismo.
#     Puede ser útil para seleccionar el tamaño de cajas anclas correcto.
#     """
#     stats = {
#         "image_height": [],
#         "image_width": [],
#         "image_mean": [],
#         "image_std": [],
#         "boxes_height": [],
#         "boxes_width": [],
#         "boxes_num": [],
#         "boxes_area": [],
#     }
#     for batch in dataset:
#         # Lote
#         x, y, x_name, y_name = batch["x"], batch["y"], batch["x_name"], batch["y_name"]

#         # Transformaciones
#         if rcnn_transform:
#             x, y = rcnn_transform([x], [y])
#             x, y = x.tensors, y[0]

#         # Entrada (imágenes)
#         stats["image_height"].append(x.shape[-2])
#         stats["image_width"].append(x.shape[-1])
#         stats["image_mean"].append(x.mean().item())
#         stats["image_std"].append(x.std().item())

#         # Objetivos (etiqueta)
#         wh = box_convert(y["boxes"], "xyxy", "xywh")[:, -2:]
#         stats["boxes_height"].append(wh[:, -2])
#         stats["boxes_width"].append(wh[:, -1])
#         stats["boxes_num"].append(len(wh))
#         stats["boxes_area"].append(box_area(y["boxes"]))

#     stats["image_height"] = torch.tensor(stats["image_height"], dtype=torch.float)
#     stats["image_width"] = torch.tensor(stats["image_width"], dtype=torch.float)
#     stats["image_mean"] = torch.tensor(stats["image_mean"], dtype=torch.float)
#     stats["image_std"] = torch.tensor(stats["image_std"], dtype=torch.float)
#     stats["boxes_height"] = torch.cat(stats["boxes_height"])
#     stats["boxes_width"] = torch.cat(stats["boxes_width"])
#     stats["boxes_area"] = torch.cat(stats["boxes_area"])
#     stats["boxes_num"] = torch.tensor(stats["boxes_num"], dtype=torch.float)

#     return stats


def from_file_to_boundingbox(file_name: pathlib.Path, groundtruth: bool = True):
    """Regresa una lista de objetos BoundingBox provenientes de una etiqueta verdadera o un predicción."""
    file = torch.load(file_name)
    labels = file["labels"]
    boxes = file["boxes"]
    scores = file["scores"] if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [
        BoundingBox(
            image_name=file_name.stem,
            class_id=l,
            coordinates=tuple(bb),
            format=BBFormat.XYX2Y2,
            bb_type=gt,
            confidence=s,
        )
        for bb, l, s in zip(boxes, labels, scores)
    ]


def from_dict_to_boundingbox(file: dict, name: str, groundtruth: bool = True):
    """Regresa una lista de objetos BoundingBox provenientes de uan etiqueta verdadera o un predicción."""
    labels = file["labels"]
    boxes = file["boxes"]
    scores = np.array(file["scores"].cpu()) if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [
        BoundingBox(
            image_name=name,
            class_id=int(l),
            coordinates=tuple(bb),
            format=BBFormat.XYX2Y2,
            bb_type=gt,
            confidence=s,
        )
        for bb, l, s in zip(boxes, labels, scores)
    ]


def log_packages_neptune(neptune_logger):
    """Usa la neptunecontrib.api para cargar los paquetes del ambiente(entorno) en uso actual."""
    dists = importlib_metadata.distributions()
    packages = {
        idx: (dist.metadata["Name"], dist.version) for idx, dist in enumerate(dists)
    }

    packages_df = pd.DataFrame.from_dict(
        packages, orient="index", columns=["package", "version"]
    )

    log_table(name="packages", table=packages_df, experiment=neptune_logger.experiment)


def log_mapping_neptune(mapping: dict, neptune_logger):
    """Usa la neptunecontrib.api para cargar un mapeo de clases."""
    mapping_df = pd.DataFrame.from_dict(
        mapping, orient="index", columns=["class_value"]
    )
    log_table(name="mapping", table=mapping_df, experiment=neptune_logger.experiment)


def log_model_neptune(
    checkpoint_path: pathlib.Path,
    save_directory: pathlib.Path,
    name: str,
    neptune_logger,
):
    """Guardar el modelo al disco local, cargarlo a neptune y removerlo de nuevo."""
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["hyper_parameters"]["model"]
    torch.save(model.state_dict(), save_directory / name)
    neptune_logger.experiment.set_property("checkpoint_name", checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(save_directory / name))
    if os.path.isfile(save_directory / name):
        os.remove(save_directory / name)


def log_checkpoint_neptune(checkpoint_path: pathlib.Path, neptune_logger):
    neptune_logger.experiment.set_property("checkpoint_name", checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(checkpoint_path))


def compute_iou(a, b):
    """Computa intersección sobre unión."""
    # obtenemos coordenadas
    xa1, ya1, wa, ha = a.T
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1, wb, hb = b.T
    xb2, yb2 = xb1 + wb, yb1 + hb
    # determinar las coordenadas (x, y) del rectangulo de intersección
    xa = torch.max(xa1, xb1)
    ya = torch.max(ya1, yb1)
    xb = torch.min(xa2, xb2)
    yb = torch.min(ya2, yb2)
    # computamos áreas
    area_a = wa * ha
    area_b = wb * hb
    # computamos intersección
    inter = torch.clamp(xb - xa + 1, min=0) * torch.clamp(yb - ya + 1, min=0)
    # computamos unión
    union = area_a + area_b - inter
    # computamos IOU
    iou = torch.mean(inter / union)
    return iou

def experiments_metric_values(session, user_project, metric, experiments=None, tags=None, legend_parameter = None, exclusive = None, smooth = [501,3]):
    """If exclusive and more than one tag look for experiments that have
     exactly all tags, if not exclusive with at least one tag is enough
     get an experiment"""
    project = session.get_project(user_project)
    if experiments:
        experiments = project.get_experiments(id=experiments)
    elif tags:
        if not exclusive:
            experiments = project.get_experiments()
            experiments = [exp for exp in experiments if any(tag in exp.get_tags() for tag in tags)]
        else:
            experiments = project.get_experiments()
            experiments = [exp for exp in experiments if all(tag in exp.get_tags() for tag in tags)]
    tot_df = {}
    tot_df_smooth = {}
    x_max = 0
    for exp in experiments:
        if len(exp.get_numeric_channels_values(metric)['x']) > x_max:
            tot_df['x'] = exp.get_numeric_channels_values(metric)['x']
            x_max = len(exp.get_numeric_channels_values(metric)['x'])
        if legend_parameter in exp.get_parameters().keys():
            tot_df[exp.get_parameters()[legend_parameter]] = exp.get_numeric_channels_values(metric)[metric]
        else:
            tot_df[exp.get_system_properties()['id']] = exp.get_numeric_channels_values(metric)[metric]
    if smooth:
      for exp in experiments:
          if legend_parameter in exp.get_parameters().keys():
              tot_df[exp.get_parameters()[legend_parameter]+'_smooth'] = pd.Series(savgol_filter(list(exp.get_numeric_channels_values(metric)[metric]),smooth[0], smooth[1]))
          else:
              tot_df[exp.id+'_smooth'] = savgol_filter(list(exp.get_numeric_channels_values(metric)[metric]),smooth[0], smooth[1])

    return pd.DataFrame.from_dict(tot_df)

def split_data(exps):
    exps_normal = exps[exps.drop('x', axis = 1).columns[:int((len(exps.columns)-1)/2)]]
    exps_normal['x'] = exps.x
    exps_smooth = exps[exps.drop('x', axis = 1).columns[int((len(exps.columns)-1)/2):]]
    exps_smooth['x'] = exps.x

    return exps_normal, exps_smooth

class LabelEncoder(object):
    """Label encoder for tag labels."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        '''Mapear etiquetas como palabras, a numeros'''
        classes = np.unique(list(itertools.chain.from_iterable(y)))
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        '''Pasar de codificación numérica a codificación OneHot'''
        y_one_hot = np.zeros((len(y), len(self.class_to_index)), dtype=int)
        for i, item in enumerate(y):
            for class_ in item:
                y_one_hot[i][self.class_to_index[class_]] = 1
        return y_one_hot

    def decode(self, y):
        '''Mapear de coficación OneHot a palabras'''
        classes = []
        for i, item in enumerate(y):
            indices = np.where(item == 1)[0]
            classes.append([self.index_to_class[index] for index in indices])
        return classes

    def save(self, fp):
        '''Guardar  el mapeo y codificaciones'''
        with open(fp, 'w') as fp:
            contents = {'class_to_index': self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)

def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which 
    'maintains balanced representation with respect 
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test
