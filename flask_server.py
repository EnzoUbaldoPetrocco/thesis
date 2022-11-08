#! /usr/bin/env python3
import json
from tracemalloc import start
from flask import Flask, request, make_response
import requests
from acquisition_evaluation import RModel
import base64, io
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import StringIO
import tensorflow as tf
import time
from PIL import Image


gpus = tf.config.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
                tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3000)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
# Virtual devices must be set before GPUs have been initialized
                print(e)
else:
        print('Using CPU')

def decrease(img, value=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = value-1
    v[v <= lim] = 0
    v[v > lim] -= value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


app = Flask(__name__)    
@app.route("/put_image", methods=["PUT"])
def put():
    start = time.time()
    req = request.get_json(force=True)
    image = req['image']
    size = req['size']
    imgdata = base64.b64decode(image)
    image_decoded = Image.frombytes("RGB", (size[1],size[0]), imgdata)
    image_decoded = cv2.cvtColor(np.array(image_decoded), cv2.COLOR_RGB2BGR)
    image_brightness_decreased = decrease(image_decoded)
    '''plt.figure()
    plt.imshow(image_decoded)
    plt.show()'''
    if model.evaluate_image(image_brightness_decreased):
        sentence = "Please turn off the Lamp" 
    else:
        sentence = "The lamp is off!"
    res= json.dumps({"sentence": sentence})
    stop = time.time()
    print("Time required: ", stop-start)
    return res

        

if __name__ == '__main__':
    lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
        4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
        2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
        1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
        4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
        2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
        1.00000000e+02]
    lamb =  lambda_grid[20]
    model = RModel("chinese",lamb )
    app.run(host="0.0.0.0", port=5000, debug=False)
   