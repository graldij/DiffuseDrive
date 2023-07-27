##segmentation backbone from https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/master
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.config import cfg
from mit_semseg.models.models import Resnet, ResnetDilated
from mit_semseg.models import resnet 
import torch
import torch.nn as nn

import sys
sys.path.append('/home/rl_course_10/DiffuseDriveM/diffuser/models')

from DeepLabV3Plus import network
from DeepLabV3Plus import utils

class FeatureExtractorCsailResnet(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        #pretrained_dir = "/scratch_net/biwidl211/rl_course_10/pretrained_model/csail_semseg"
        #cfg.merge_from_file(pretrained_dir + "/config/ade20k-resnet50-upernet.yaml")
        #cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        #cfg.MODEL.weights_encoder = os.path.join(
        #    pretrained_dir + "/" + cfg.DIR, 'encoder_' + cfg.TEST.checkpoint
        #)
        raise NotImplementedError

        orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
        net_encoder = Resnet(orig_resnet)
        self.backbone = net_encoder

        if freeze:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.backbone(x, return_feature_maps=True)
        return x

class FeatureExtractorDeeplabMobilenet(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes= 19, output_stride= 16)
        # network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        checkpoint = torch.load("/scratch_net/biwidl211/rl_course_10/pretrained_model/deeplabv3/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model.backbone)
        model.to("cuda")
        del checkpoint
        self.backbone = model
        print(self.backbone)

        if freeze:
            self.backbone.eval()
            
        self.conv_reduce = nn.Conv2d(320, 20, 1, padding='same')
    
    def forward(self, x):
        x = self.backbone(x)['out']
        x = self.conv_reduce(x)
        return x