#Libraries
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
import sys
from PIL import Image
import torch
import torchvision

import argparse
from distutils.version import LooseVersion

from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from mit_semseg.config import cfg

import yacs


#image_path or image object is given by the argument of function



#transformation funcion for input image
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])


num_epoch = 'epoch_20.pth'
colors = scipy.io.loadmat('./data/color19.mat')['colors']
num_classes = 19
#input_image = image_path

#CODE FOR SEGM===========================================================
assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'


cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

# absolute paths of model weights
cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + num_epoch)
cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + num_epoch)

cfg.DATASET.num_class = num_classes

assert os.path.exists(cfg.MODEL.weights_encoder) and os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

#create encoder decoder and other stuff for prediction
# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder,
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)
net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder,
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()
#==========================================================================


def create_pred(input_image):
    #import input image
    #pil_image = PIL.Image.open(input_image)
    img_data = pil_to_tensor(input_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    #Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred


#function to visualize
def color_image(pred):
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)
    # aggregate images and save
    im = PIL.Image.fromarray(pred_color)
    return im