#! /usr/bin/env python3

from classificator_featureextraction import SVCClassificator
kernel = [ 'linear', 'rbf' ]
ds_selections = ['chinese', 'french', 'mix']
linear_mix_svc_classificator = SVCClassificator(ds_selections[2])
linear_mix_svc_classificator.ds_selection = ds_selections[2]
linear_mix_svc_classificator.kernel = kernel[0]
linear_mix_svc_classificator.execute()