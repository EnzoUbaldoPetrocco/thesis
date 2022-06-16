#! /usr/bin/env python3

from classificator import SVCClassificator

kernel = [ 'linear', 'rbf' ]
ds_selections = ['chinese', 'french', 'mix']

linear_french_svc_classificator = SVCClassificator()
linear_french_svc_classificator.ds_selection = ds_selections[1]
linear_french_svc_classificator.kernel = kernel[0]
linear_french_svc_classificator.execute()

linear_chinese_svc_classificator = SVCClassificator()
linear_chinese_svc_classificator.ds_selection = ds_selections[0]
linear_chinese_svc_classificator.kernel = kernel[0]
linear_chinese_svc_classificator.execute()

linear_mix_svc_classificator = SVCClassificator()
linear_mix_svc_classificator.ds_selection = ds_selections[2]
linear_mix_svc_classificator.kernel = kernel[0]
linear_mix_svc_classificator.execute()

gaussian_french_svc_classificator = SVCClassificator()
gaussian_french_svc_classificator.ds_selection = ds_selections[1]
gaussian_french_svc_classificator.kernel = kernel[1]
gaussian_french_svc_classificator.execute()

gaussian_chinese_svc_classificator = SVCClassificator()
gaussian_chinese_svc_classificator.ds_selection = ds_selections[0]
gaussian_chinese_svc_classificator.kernel = kernel[1]
gaussian_chinese_svc_classificator.execute()

gaussian_mix_svc_classificator = SVCClassificator()
gaussian_mix_svc_classificator.ds_selection = ds_selections[2]
gaussian_mix_svc_classificator.kernel = kernel[1]
gaussian_mix_svc_classificator.execute()