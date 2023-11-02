# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:38:11 2022

@author: 38688
"""

import numpy as np
from make_case import case_2
import pandas as pd

case = case_2()
capacity = np.array(pd.read_excel('case_l.xlsx', sheet_name=0),dtype=float)
case.exist_capacity[:,1] = np.mean(capacity[-1000:,7:13],axis=0)
record = np.zeros((2000,10))
for i in range(20):
    record[i*case.scene_num:(i+1)*case.scene_num,0], record[i*case.scene_num:(i+1)*case.scene_num,1], \
        record[i*case.scene_num:(i+1)*case.scene_num,2], record[i*case.scene_num:(i+1)*case.scene_num,3], \
            record[i*case.scene_num:(i+1)*case.scene_num,4], record[i*case.scene_num:(i+1)*case.scene_num,5], \
                record[i*case.scene_num:(i+1)*case.scene_num,6], record[i*case.scene_num:(i+1)*case.scene_num,7], \
                    record[i*case.scene_num:(i+1)*case.scene_num,8], \
                        record[i*case.scene_num:(i+1)*case.scene_num,9]= case.get_final_information()
data = pd.DataFrame(record)
writer = pd.ExcelWriter('data_l.xlsx')
data.to_excel(writer, 'page_1')
writer.save()
writer.close()