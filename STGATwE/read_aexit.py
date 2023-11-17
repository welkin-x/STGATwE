# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:50:55 2023

@author: 22114
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os


# In[]
def read_ex_txt(path,station_hex):  #此处目前仅能读ex
	# 从文件中读取内容，sep是要选用的分割符
    path_list = os.listdir(path)
    all_data = [] 
    for i in range(len(path_list)):
        txt_file_path = os.path.join(path,path_list[i])
        one_data = pd.read_table(txt_file_path, sep='\t',engine='python')
        one_data = one_data.iloc[:-3,:]
        one_data = one_data.loc[:,['CARDID(varchar(100))','ENSTATIONHEX(varchar(20))','ENTRYLANE(int(11))','ENTRYTIME(datetime)','EXITSTATION(int(11))','EXITLANE(int(11))','EXSTATIONHEX(varchar(20))','EXITTIME(datetime)']]
        one_data = one_data[one_data['EXSTATIONHEX(varchar(20))'].isin(station_hex)]
        one_data['EXITSTATION(char)'] =  one_data['EXITSTATION(int(11))'].astype(str)
        all_data.append(one_data)
    print(1)
    data_frame_concat = pd.concat(all_data,axis=0,ignore_index=True)
    print(2)
    #将data分别存入字典，即Image对应count
    return data_frame_concat
    
def sort_station(data):
    en_station_list = []
    ex_station_list = []
    for index,row in data.iterrows():
        if row['ENSTATIONHEX(varchar(20))'] not in en_station_list:
            en_station_list.append(row['ENSTATIONHEX(varchar(20))'])
        if row['EXSTATIONHEX(varchar(20))'] not in ex_station_list:
            ex_station_list.append(row['EXSTATIONHEX(varchar(20))'])
    # print("en_station_list:",en_station_list)
    # print("ex_station_list:",ex_station_list)
    # print("en_len:",len(en_station_list))
    # print("ex_len:",len(ex_station_list))
    return en_station_list,ex_station_list

if __name__ == '__main__':
    #path = r'F:\trial_aentry'
    path = '.\data_one_day_aexit'
    station = [2.11e+06,3201211003,3201211004,3201211005,3201211006] #0,2,3,4,5,6
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    data = read_ex_txt(path,station_hex)
# In[]


#412
