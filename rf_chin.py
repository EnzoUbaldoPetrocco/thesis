#! /usr/bin/env python3

from RFC import RFCClassificator
ds_selections = ['chinese', 'french', 'mix']
gaussian_chinese_svc_classificator = RFCClassificator()
gaussian_chinese_svc_classificator.ds_selection = ds_selections[0]
gaussian_chinese_svc_classificator.execute()
