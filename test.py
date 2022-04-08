import requests
import matplotlib.pyplot as plt
import json
import numpy as np
import sys

# resp = requests.post("http://localhost:5000/addimg",
#                      files={"file": open('./images/cat.jpg','rb')})
# print(resp.json())
sys.path.append('..')
from isegm.utils import vis 
from isegm.inference.clicker import Click
resp_img = requests.get("http://localhost:5000/getimg")
img_arr = np.array(json.loads(resp_img.json()))


resp_mask = requests.post('http://localhost:5000/click', json = {
    "click":{
        "is_postive": True,
        "coords_x": 220,
        "coords_y": 160
    } 
})
init_clicks = [Click(is_positive=True, coords=(220, 160))]
pred_mask = np.array(json.loads(resp_mask.json()))

draw = vis.draw_with_blend_and_clicks(img_arr, mask=pred_mask, clicks_list=init_clicks)
draw = np.concatenate((draw,
    255 * pred_mask[:, :, np.newaxis].repeat(3, axis=2)
), axis=1)

plt.figure(figsize=(20, 30))
plt.imshow(draw)
plt.show()