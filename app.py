from re import T
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
import torch
from isegm.inference import utils
from isegm.inference.predictors import get_predictor
from flask_cors import CORS
from PIL import Image
import io
import requests
import pickle

device = torch.device('cpu')

EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49

app = Flask(__name__)
CORS(app)
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
checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH_FOR_WEB, 'coco_lvis_h18s_itermask')
model = utils.load_is_model(checkpoint_path, device)

brs_mode = 'f-BRS-B'
predictor = get_predictor(model, brs_mode, device, prob_thresh=MODEL_THRESH)

annotating_image = image
predictor.set_input_image(annotating_image)
clicks_list = []
pred_mask = None


@app.route('/click', methods=['POST'])
def click():
    global clicks_list
    global pred_mask
    if request.method == 'POST':
        # we will get the file from the request
        
        data = request.get_json()
        is_positive = data['click']['is_postive']
        click_info_x = data['click']['coords_x']
        click_info_y = data['click']['coords_y']
        # convert that to bytes
        #print("click info:", is_positive, click_info_x, click_info_y)
        click = Click(is_positive=is_positive, coords=(click_info_y, click_info_x))
        clicks_list.append(click)
        single_click = [click]
        
        _, _, pred_mask = get_prediction_from_click(predictor, single_click=single_click)
        #print(pred_mask[click_info_y, click_info_x])
        #print(pred_mask[click_info_x-10:click_info_x+10, click_info_y-10:click_info_y+10])
        draw = vis.draw_with_blend_and_clicks(annotating_image, mask=pred_mask, clicks_list=clicks_list)
        # draw = np.concatenate((draw,
        #     255 * pred_mask[:, :, np.newaxis].repeat(3, axis=2)
        # ), axis=1)
        # print(annotating_image[420:430, 350:370])
        # print("--------------------------")
        # print(draw[420:430, 350:370])
        img_arr = Image.fromarray(np.uint8(draw))
        im_byte = io.BytesIO()
        img_arr.save(im_byte, format='JPEG')
        my_string = str(base64.b64encode(im_byte.getvalue()))
        response = jsonify({"img": my_string[2:-1]})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/train', methods=['POST'])
def train_single():
    global annotating_image
    global pred_mask
    data = request.get_json()
    label = data['label']
    img_bytes = pickle.dumps(annotating_image)
    mask_bytes = pickle.dumps(pred_mask)
    resp = requests.post("http://localhost:7000/trigger_finetune",
                      files={"file": img_bytes, "mask": mask_bytes, "label":label})
    print(resp.json())
    return resp.json()

@app.route('/addimg', methods=['POST'])
def add_img():
    global annotating_image
    global clicks_list
    global pred_mask
    pred_mask = None
    clicks_list = []
    data = request.get_data()
    img = base64.decodebytes(data[22:])
    annotating_image = np.array(transform_image(img))

    predictor.set_input_image(annotating_image)
    response = jsonify({'status': "success"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/getimg', methods=['GET'])
def get_annotating_img():
    #print(annotating_image)
    im = Image.fromarray(annotating_image)
    im_byte = io.BytesIO()
    im.save(im_byte, format='JPEG')
    my_string = str(base64.b64encode(im_byte.getvalue()))
    response = jsonify({"img": my_string[2:-1]})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def main():
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()