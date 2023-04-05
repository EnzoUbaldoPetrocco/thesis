#! /usr/bin/env python3

from classificator import SVCClassificator
kernel = [ 'linear', 'rbf' ]
ds_selections = ['chinese', 'french', 'mix']
linear_french_svc_classificator = SVCClassificator()
linear_french_svc_classificator.ds_selection = ds_selections[1]
linear_french_svc_classificator.kernel = kernel[0]
linear_french_svc_classificator.execute()