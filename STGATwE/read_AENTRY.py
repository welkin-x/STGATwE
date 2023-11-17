# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format',lambda x:'%.2f'%x)
import numpy as np
import math
import os


# In[]
def read_en_txt(path,station,station_hex):
	# 从文件中读取内容，sep是要选用的分割符
    path_list = os.listdir(path)
    all_data = [] 
    for i in range(len(path_list)):
        txt_file_path = os.path.join(path,path_list[i])
        one_data = pd.read_table(txt_file_path, sep='\t',engine='python')
        one_data = one_data.iloc[:-3,:]
        one_data = one_data.loc[:,['CARDID(varchar(40))','ENTRYSTATION(int(11))','ENTRYLANE(int(11))','ENTRYTIME(datetime)']]
        one_data['ENTRYSTATION(char)'] =  one_data['ENTRYSTATION(int(11))'].astype(str)
        one_data = one_data[one_data['ENTRYSTATION(char)'].isin(station)]
        all_data.append(one_data)
    print(1)
    data_frame_concat = pd.concat(all_data,axis=0,ignore_index=True)
    print(2)
    #将data分别存入字典，即Image对应count
    data_frame_concat['EXSTATIONHEX(varchar(20))']=data_frame_concat['ENTRYSTATION(char)'].replace(station,station_hex)
    return data_frame_concat
    
    
if __name__ == '__main__':
    #path = r'F:\trial_aentry'
    path = '.\data_one_day_aentry'
    # path_list = os.listdir(path)
    station = ['2110002.0','2110003.0','2110004.0','2110005.0','2110006.0'] #0,2,3,4,5,6
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    data = read_en_txt(path,station,station_hex)
# In[]
    
    # station_list = []
    # for index,row in data.iterrows():
    #     if int(row['ENTRYSTATION(int(11))']) not in station_list:
    #         station_list.append(int(row['ENTRYSTATION(int(11))']))
    # print(station_list)
    # print(len(station_list))

#412