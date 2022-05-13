import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from copy import deepcopy
import torchvision.transforms as transforms

sys.path.insert(0, '..')
from isegm.utils import vis, exp
import io
from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.inference.predictors import get_predictor

EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49

device = torch.device('cpu')
cfg = exp.load_config_file('./web_config.yml', return_edict=True)

def transform_image(image_bytes):
    # my_transforms = transforms.Compose([transforms.Resize(255)
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(
    #                                         [0.485, 0.456, 0.406],
    #                                         [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    rgb_im = image.convert('RGB')
    return rgb_im



def get_prediction_from_click(predictor, single_click=None, pred_thr=MODEL_THRESH):
    clicker = Clicker(init_clicks=single_click)
    with torch.no_grad():
        pred_probs = predictor.get_prediction(clicker)
        pred_mask = pred_probs > pred_thr
    ious_list = [] 
    
    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_mask


