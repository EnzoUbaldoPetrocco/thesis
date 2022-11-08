import os
import qi
import argparse
import time
import naoqi
from naoqi import ALProxy
import sys
import json
reload(sys)
app_name = "lamps"
sys.path.append("/data/home/nao/.local/share/PackageManager/apps/" + app_name + "/libs/")
sys.path.append("/data/home/nao/.local/share/PackageManager/apps/" + app_name + "/")

class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)
        #session = qi.Session()
        self.path = ALFrameManager.getBehaviorPath(self.behaviorId)
        self.motion = self.session().service('ALMotion')
        #self.photo_capture = self.session().service( "ALPhotoCapture" )
        self.photo_capture = ALProxy("ALPhotoCapture")
        self.image_path = self.path + "/image.png"
        server_ip = "130.251.13.135"
        nome_servizio = "put_image"
        self.request_uri = "http://" + server_ip + ":5000/" + nome_servizio
        self.TTS = self.session().service('ALTextToSpeech')

    def onLoad(self):
        #put initialization code here
        pass

    def onUnload(self):
        #put clean-up code here
        pass

    def onInput_onStart(self):
        #self.onStopped() #activate the output of the box
        pass

    def onInput_onStop(self):
        for i in range(0,5):
            for i in range(0,4):
                self.motion.moveTo(0,0,3.14/2)
                self.motion.moveTo(0.5,0,0)
            self.photo_capture.setPictureFormat("png")
            self.photo_capture.takePictures(1,self.image_path, "image")
            image = cv2.imread(self.image_path)
            size = image.shape
            image = image.tobytes()
            image_encoded = base64.b64encode(image)
            msg = {'image': image_encoded, 'size': size}
            req = json.dumps(msg)
            res=requests.put(url, data=req, verify=False)
            sentence = res.json()["sentence"]
            self.TTS.say(sentence)
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box