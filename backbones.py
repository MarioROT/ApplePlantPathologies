import torch
import torchvision.models as models
from torch import nn
# from torchvision.models import efficientnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
# import torchxrayvision as xrv


# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.
class GetModel(nn.Module):
    def __init__(self, n_classes, backbone_net, freeze = False):
        super().__init__()
        mod, in_channs = get_backbone(backbone_net)
        if freeze:
          for i, param in enumerate(mod.parameters()):
            param.requires_grad = False
        if 'mobilenet' in backbone_net or 'efficientnet' in backbone_net or 'densenet' in backbone_net:
            mod.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=in_channs, out_features=n_classes)
            )
        else:
            mod.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=in_channs, out_features=n_classes)
            )
        self.base_model = mod
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

class GetTorchXrayVisionModel(nn.Module):
    def __init__(self, n_classes, backbone_net, freeze=False):
      super().__init__()
      # mod_xrv = xrv.models.get_model(backbone_net)
      mod_xrv =  xrv.models.DenseNet(weights=backbone_net)
      if freeze:
        for i, param in enumerate(mod_xrv.parameters()):
          param.requires_grad = False
      mod_xrv.op_threshs = None
      mod_xrv.classifier = nn.Sequential(
          # nn.Dropout(p=0.2),
          nn.Linear(in_features=mod_xrv.classifier.in_features, out_features=n_classes),
      )
      self.base_model = mod_xrv
      self.sigm = nn.Sigmoid()

    def forward(self, x):
      return self.base_model(x)

# ---------------------- Models for new metrics experiments ------------------ #

class GetModelNM(nn.Module):
    def __init__(self, n_classes, backbone_net, freeze, i_weights):
        super().__init__()
        mod,in_channs = get_backbone(backbone_net)
        if i_weights:
          self.apply(self._init_weights)
        if freeze:
          for param in mod.parameters():
              param.requires_grad = False
        if 'mobilenet' in backbone_net or 'efficientnet' in backbone_net or 'densenet' in backbone_net:
            mod.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=in_channs, out_features=n_classes)
            )
        else:
            mod.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=in_channs, out_features=n_classes)
            )
        self.base_model = mod

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x):
        return self.base_model(x)

def build_backbone(model):

    try:
        arch, package, weights = model.split('-')

        if weights == 'none':
            weights = None
        else:
            if weights == 'mimic':
                weights = 'mimic_nb'
            weights = f'densenet121-res224-{weights}'

        backbone = xrv.models.DenseNet(weights=weights)
        backbone.op_threshs = None
        backbone.classifier = nn.Identity()
        backbone.output_size = 1024
        print('TorchXrayVision Model Loaded!')
        return backbone

    except:
        print('Model {model} not available, a densenet121 Imagenet pretrained provided instead')
        backbone = models.densenet121(weights='DEFAULT')
        backbone.classifier = nn.Identity()
        backbone.output_size = 1024
        return backbone

class BuildModel(nn.Module):
    def __init__(self, n_classes, backbone_net, freeze):
        super().__init__()
        self.features = build_backbone(backbone_net)
        self.fc = nn.Linear(self.features.output_size,n_classes)
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        # self.features.classifier = nn.Linear(self.features.output_size,8)

    def forward(self, x):
        return self.fc(self.features(x))

# ---------------------------------------------------------------------------- #

def get_backbone(backbone_name: str):
    """
    Regresa una arquitectura base versión de EfficientNet, ShuffleNet v2,
    MobileNet v2 o v3, EfficientNet pre-entrenada en ImageNet.
    """

    if backbone_name == "mobilenet_v2":
        pretrained_model = models.mobilenet_v2(weights = 'DEFAULT', progress=False)
        out_channels = 1280
    elif backbone_name == "mobilenet_v3":
        pretrained_model = models.mobilenet_v3_large(weights = 'DEFAULT', progress=False)
        out_channels = 1280
    elif backbone_name == "resnet18":
        pretrained_model = models.resnet18(weights = 'DEFAULT', progress=False)
        out_channels = 512
    elif backbone_name == "resnet34":
        pretrained_model = models.resnet34(weights = 'DEFAULT', progress=False)
        out_channels = 512
    elif backbone_name == "resnet50":
        pretrained_model = models.resnet50(weights = 'DEFAULT', progress=False)
        out_channels = 2048
    elif backbone_name == "resnet101":
        pretrained_model = models.resnet101(weights = 'DEFAULT', progress=False)
        out_channels = 2048
    elif backbone_name == "resnet152":
        pretrained_model = models.resnet152(weights = 'DEFAULT', progress=False)
        out_channels = 2048
    elif backbone_name == "resnext50_32x4d":
        pretrained_model = models.resnext50_32x4d(weights = 'DEFAULT')
        out_channels = 2048
    elif backbone_name == "shufflenet_v2_x0_5":
        pretrained_model = models.shufflenet_v2_x0_5(weights = 'DEFAULT', progress=False)
        out_channels = 1024
    elif backbone_name == "shufflenet_v2_x1_0":
        pretrained_model = models.shufflenet_v2_x1_0(weights = 'DEFAULT', progress=False)
        out_channels = 1024
    elif backbone_name == "shufflenet_v2_x1_5":
        pretrained_model = models.shufflenet_v2_x1_5(weights = 'DEFAULT', progress=False)
        out_channels = 1024
    elif backbone_name == "shufflenet_v2_x2_0":
        pretrained_model = models.shufflenet_v2_x2_0(weights = 'DEFAULT', progress=False)
        out_channels = 2048
    elif backbone_name == "efficientnet_b0":
        pretrained_model = models.efficientnet_b0(weights = 'DEFAULT', progress=False)
        out_channels = 1280
    elif backbone_name == "efficientnet_b1":
        pretrained_model = models.efficientnet_b1(weights = 'DEFAULT', progress=False)
        out_channels = 1280
    elif backbone_name == "efficientnet_b2":
        pretrained_model = models.efficientnet_b2(weights = 'DEFAULT', progress=False)
        out_channels = 1408
    elif backbone_name == "efficientnet_b3":
        pretrained_model = models.efficientnet_b3(weights = 'DEFAULT', progress=False)
        out_channels = 1536
    elif backbone_name == "efficientnet_b4":
        pretrained_model = models.efficientnet_b4(weights = 'DEFAULT', progress=False)
        out_channels = 1792
    elif backbone_name == "efficientnet_b5":
        pretrained_model = models.efficientnet_b5(weights = 'DEFAULT', progress=False)
        out_channels = 2048
    elif backbone_name == "efficientnet_b6":
        pretrained_model = models.efficientnet_b6(weights = 'DEFAULT', progress=False)
        out_channels = 2304
    elif backbone_name == "efficientnet_b7":
        pretrained_model = models.efficientnet_b7(weights = 'DEFAULT', progress=False)
        out_channels = 2560
    elif backbone_name == "densenet121":
        pretrained_model = models.densenet121(weights = 'DEFAULT', progress=False)
        out_channels = 1024
    elif backbone_name == "densenet161":
        pretrained_model = models.densenet161(weights = 'DEFAULT', progress=False)
        out_channels = 2208
    elif backbone_name == "densenet169":
        pretrained_model = models.densenet169(weights = 'DEFAULT', progress=False)
        out_channels = 1664
    elif backbone_name == "densenet201":
        pretrained_model = models.densenet201(weights = 'DEFAULT', progress=False)
        out_channels = 1920

    return pretrained_model, out_channels #backbone

