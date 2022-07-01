#! /usr/bin/env python3

from mclassificator import SVCClassificator
kernel = [ 'linear', 'rbf' ]
ds_selections = ['chinese', 'french']
gaussian_chinese_svc_classificator = SVCClassificator()
gaussian_chinese_svc_classificator.ds_selection = ds_selections[1]
gaussian_chinese_svc_classificator.kernel = kernel[1]
gaussian_chinese_svc_classificator.execute()