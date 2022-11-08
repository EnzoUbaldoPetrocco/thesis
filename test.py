#! /usr/bin/env python3
from acquisition_evaluation import RModel
from RobustResNet50 import FeatureExtractor
import cv2
rebuild_model = True


lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
        4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
        2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
        1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
        4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
        2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
        1.00000000e+02]
lamb =  lambda_grid[20]

if rebuild_model == True:
        chinese_model = FeatureExtractor("chinese", lamb=lamb)
        french_model = FeatureExtractor("french", lamb=lamb)

chin21 = RModel("chinese", lamb)
chinese = cv2.imread("./chinese.png")
chinese_on = cv2.imread("./chineseon.jpg")
french = cv2.imread("./french.png")
french_on = cv2.imread("./frenchon.jpg")



res = chin21.evaluate_image(chinese)
print("Chinese off lamp is " + str(res))
res = chin21.evaluate_image(chinese_on)
print("Chinese on lamp is " + str(res))
res = chin21.evaluate_image(french)
print("French off lamp is " + str(res))
res = chin21.evaluate_image(french_on)
print("French on lamp is " + str(res))
