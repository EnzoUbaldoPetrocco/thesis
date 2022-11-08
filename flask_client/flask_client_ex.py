#! /usr/bin/env python
import json
from flask import Flask, request, make_response
import requests
import base64
import cv2
app = Flask(__name__)

############################################
#########  CHINESE OFF ####################
ip = "130.251.13.135"
url = "http://" + ip + ":5000/" + "put_image"
image = cv2.imread('../chinese.png')
size = image.shape
image = image.tobytes()
image_encoded = base64.b64encode(image)
msg = {'image': image_encoded, 'size': size}
req = json.dumps(msg)
headers = {'content-type': 'application/json'}
res=requests.put(url, data=req, verify=False)
sentence = res.json()["sentence"]
print('sentence:')
print(sentence)

############################################
#########  CHINESE ON ####################
image = cv2.imread('../chineseon.png')
size = image.shape
image = image.tobytes()
image_encoded = base64.b64encode(image)
msg = {'image': image_encoded, 'size': size}
req = json.dumps(msg)
headers = {'content-type': 'application/json'}
res=requests.put(url, data=req, verify=False)
sentence = res.json()["sentence"]
print('sentence:')
print(sentence)

############################################
#########  FRENCH OFF ####################
image = cv2.imread('../french.png')
size = image.shape
image = image.tobytes()
image_encoded = base64.b64encode(image)
msg = {'image': image_encoded, 'size': size}
req = json.dumps(msg)
headers = {'content-type': 'application/json'}
res=requests.put(url, data=req, verify=False)
sentence = res.json()["sentence"]
print('sentence:')
print(sentence)

############################################
#########  FRENCH ON ####################
image = cv2.imread('../frenchon.png')
size = image.shape
image = image.tobytes()
image_encoded = base64.b64encode(image)
msg = {'image': image_encoded, 'size': size}
req = json.dumps(msg)
headers = {'content-type': 'application/json'}
res=requests.put(url, data=req, verify=False)
sentence = res.json()["sentence"]
print('sentence:')
print(sentence)