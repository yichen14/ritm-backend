from urllib import request
from flask import jsonify
from flask import Flask, request
import matplotlib.pyplot as plt
import json
import numpy as np
from handle_click import get_prediction_from_click, transform_image
from isegm.inference.clicker import Click
import cv2
from isegm.utils import vis, exp
import base64


EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49

app = Flask(__name__)
cfg = exp.load_config_file('./web_config.yml', return_edict=True)
# from isegm.inference import utils
# DATASET = 'GrabCut'
# dataset = utils.get_dataset(DATASET, cfg)


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
image_path = cfg.TEST_IMAGES
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


annotating_image = image

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
        clicks_list, ious_arr, pred = get_prediction_from_click(annotating_image, single_click=single_click)
        pred_mask = pred > MODEL_THRESH
        return jsonify(json.dumps(pred_mask.tolist()))

@app.route('/addimg', methods=['POST'])
def add_img():
    data = request.files['file'].read()
    annotating_image = transform_image(data)
    return jsonify({'status': "success"})

@app.route('/getimg', methods=['GET'])
def get_annotating_img():
    
    return jsonify(json.dumps(annotating_image.tolist()))