# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:35:21 2023

@author: 22114
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os



# In[]
def read_txt(txt_file_path):
	# 从文件中读取内容，sep是要选用的分割符
    data = pd.read_table(txt_file_path, sep='\t',engine='python')
    #将data分别存入字典，即Image对应count
    return data
    
    
if __name__ == '__main__':
    file= r"F:\AENTRY\RAW_AENTRY_20200901000000.txt"
    data = read_txt(file)
    file2= r"F:\AENTRY\RAW_AENTRY_20200901000500.txt"
    data2 = read_txt(file2)
    data = data.iloc[:-3,:]
    data2 = data2.iloc[:-3,:]
    data2 = data2.loc[:,['CARDID(varchar(40))','ENTRYSTATION(int(11))','ENTRYLANE(int(11))','ENTRYTIME(datetime)']]
# In[]
station_list = []
for row in range(data.shape[0]):
    if int(data.iloc[row]['ENTRYSTATION(int(11))']) not in station_list:
        print(data.iloc[row]['ENTRYSTATION(int(11))'])
        station_list.append(int(data.iloc[row]['ENTRYSTATION(int(11))']))
print(station_list)
print(len(station_list))
# In[]
station_list = []
for row in range(data2.shape[0]):
    if int(data2.iloc[row]['ENTRYSTATION(int(11))']) not in station_list:
        print(data2.iloc[row]['ENTRYSTATION(int(11))'])
        station_list.append(int(data2.iloc[row]['ENTRYSTATION(int(11))']))
print(station_list)
print(len(station_list))