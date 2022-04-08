import requests
import matplotlib.pyplot as plt
import json
import numpy as np
import sys


sys.path.append('..')
from isegm.utils import vis 
from isegm.inference.clicker import Click

# resp = requests.post("http://localhost:5000/addimg",
#                      files={"file": open('./images/cat2.jpg','rb')})
# print(resp.json())

resp_img = requests.get("http://localhost:5000/getimg")
img_arr = np.array(json.loads(resp_img.json()))
clicks_list = []

def test_single_click_n_visiualize(coords_x, coords_y, is_postive = True):
    resp_mask = requests.post('http://localhost:5000/click', json = {
        "click":{
            "is_postive": is_postive,
            "coords_x": coords_x,
            "coords_y": coords_y
        } 
    })
    clicks_list.append(Click(is_positive=True, coords=(coords_x, coords_y)))
    pred_mask = np.array(json.loads(resp_mask.json()))
    draw = vis.draw_with_blend_and_clicks(img_arr, mask=pred_mask, clicks_list=clicks_list)
    draw = np.concatenate((draw,
        255 * pred_mask[:, :, np.newaxis].repeat(3, axis=2)
    ), axis=1)

    plt.figure(figsize=(20, 30))
    plt.imshow(draw)
    plt.show()

test_single_click_n_visiualize(800, 1250)
test_single_click_n_visiualize(220, 160)
test_single_click_n_visiualize(429, 360)
test_single_click_n_visiualize(1250, 800, False)


# test_single_click_n_visiualize(700, 1050)