#! /usr/bin/env python3
from FeatureExtractor_ResNet50 import FeatureExtractor

lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
        4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
        2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
        1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
        4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
        2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
        1.00000000e+02]
lamb =  lambda_grid[8]

chinese_model = FeatureExtractor("chinese", lamb=lamb)
french_model = FeatureExtractor("french", lamb=lamb)

