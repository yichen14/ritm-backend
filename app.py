from urllib import request
from flask import jsonify
from flask import Flask, request
import matplotlib.pyplot as plt
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from copy import deepcopy
from IPython import embed
import torchvision.transforms as transforms
sys.path.insert(0, '..')
from isegm.utils import vis, exp
import io
from isegm.inference import utils

device = torch.device('cuda:0')
cfg = exp.load_config_file('../config.yml', return_edict=True)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)



from isegm.inference.clicker import Click
from isegm.inference.clicker import Clicker
def get_prediction_from_click(image, predictor, max_iou_thr, pred_thr=0.49, single_click=None):

    clicker = Clicker(init_clicks=single_click)
    with torch.no_grad():
        predictor.set_input_image(image)
        pred_probs = predictor.get_prediction(clicker)
        pred_mask = pred_probs > pred_thr
    ious_list = [] 
    
    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

from isegm.inference.predictors import get_predictor

EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49

checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH_FOR_WEB, 'coco_lvis_h18s_itermask')
model = utils.load_is_model(checkpoint_path, device)

brs_mode = 'f-BRS-B'
predictor = get_predictor(model, brs_mode, device, prob_thresh=MODEL_THRESH)



app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'



DATASET = 'GrabCut'
dataset = utils.get_dataset(DATASET, cfg)


'''
    request sample:
{
    "click":{
        "is_postive": true,
        "coords_x": 220,
        "coords_y": 160
    } 
}
'''

annotating_image = dataset.get_sample(12).image

@app.route('/click', methods=['POST'])
def click():
    if request.method == 'POST':
        # we will get the file from the request
        TARGET_IOU = 0.95
        data = request.get_json()
        is_positive = data['click']['is_postive']
        click_info_x = data['click']['coords_x']
        click_info_y = data['click']['coords_y']
        # convert that to bytes

        single_click = [Click(is_positive=is_positive, coords=(click_info_x, click_info_y))]
        clicks_list, ious_arr, pred = get_prediction_from_click(annotating_image, predictor, 
                                              pred_thr=MODEL_THRESH, 
                                              max_iou_thr=TARGET_IOU,
                                              single_click=single_click)
        pred_mask = pred > MODEL_THRESH
        print(pred_mask)
        return jsonify({'pred_mask': ""})

@app.route('/addimg', methods=['POST'])
def add_img():
    data = request.files['file'].read()
    annotating_image = transform_image(data)
    return jsonify({'status': "success"})
