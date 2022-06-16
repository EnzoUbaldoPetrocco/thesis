#! /usr/bin/env python3

from classificator import SVCClassificator
kernel = [ 'linear', 'rbf' ]
ds_selections = ['chinese', 'french', 'mix']
linear_chinese_svc_classificator = SVCClassificator()
linear_chinese_svc_classificator.ds_selection = ds_selections[0]
linear_chinese_svc_classificator.kernel = kernel[0]
linear_chinese_svc_classificator.execute()