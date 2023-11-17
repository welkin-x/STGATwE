# -*- coding: utf-8 -*-
from read_aexit import read_ex_txt 
from read_aexit import sort_station
from read_AENTRY import read_en_txt
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import seaborn as sns
import pandas as pd
import os
start_time_date = '2020-09-01 00:00:00'
# end_time = '2020-10-01 00:00:00'
end_time_date = '2020-10-01 00:00:00'
delta = datetime.timedelta(minutes=15)
date_start = datetime.datetime(2020, 9, 1)
data_end = datetime.datetime(2020, 10, 1)
start_time =  int(time.mktime(time.strptime(start_time_date, "%Y-%m-%d %H:%M:%S")))
end_time =  int(time.mktime(time.strptime(end_time_date, "%Y-%m-%d %H:%M:%S")))
time_interval = 15 * 60 #15min

def get_ex_station_flow(path,station_hex):  #计算出口站每time_interval分钟的流量
    data = read_ex_txt(path,station_hex)
    ex_station_flow = np.zeros([len(station_hex),(end_time-start_time)//time_interval])  #根据时间戳划分时间区间
    dict_ex_station_flow = dict(zip(station_hex,ex_station_flow))
    for index,row in data.iterrows():
        now_time = int(time.mktime(time.strptime(row['EXITTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
        if now_time >= start_time and row['EXSTATIONHEX(varchar(20))'] in station_hex:
            dict_ex_station_flow[row['EXSTATIONHEX(varchar(20))']][(now_time - start_time)//time_interval]=dict_ex_station_flow[row['EXSTATIONHEX(varchar(20))']][(now_time - start_time)//time_interval]+1
    return dict_ex_station_flow , data       
def get_en_station_flow(path,station,station_hex):  #计算每五分钟的流量
    data = read_en_txt(path,station,station_hex)
    en_station_flow = np.zeros([len(station_hex),(end_time-start_time)//time_interval])  #根据时间戳划分时间区间
    dict_en_station_flow = dict(zip(station_hex,en_station_flow))
    for index,row in data.iterrows():
        now_time = int(time.mktime(time.strptime(row['ENTRYTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
        #实际上将ENstationhex记为了ex 但是没有影响
        if now_time >= start_time and row['EXSTATIONHEX(varchar(20))'] in station_hex:
            dict_en_station_flow[row['EXSTATIONHEX(varchar(20))']][(now_time - start_time)//time_interval]=dict_en_station_flow[row['EXSTATIONHEX(varchar(20))']][(now_time - start_time)//time_interval]+1
    return dict_en_station_flow , data       
def get_travel_time_interval(data , station_hex):  #需传入ex读出的data
    '''仅得到流量
    travel_time = np.zeros([len(station_hex),len(station_hex),(end_time-start_time)//time_interval])
    for index,row in data.iterrows(): 
        now_time = int(time.mktime(time.strptime(row['EXITTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
        o_station_index = station_hex.index(row['ENSTATIONHEX(varchar(20))'])
        d_station_index = station_hex.index(row['EXSTATIONHEX(varchar(20))'])
        if now_time >= start_time:
            travel_time[o_station_index][d_station_index][(now_time - start_time)//time_interval] += 1
        return travel_time
    '''
    '''
    返回一个时间划分后的旅行时间矩阵，列表shape为len(station_hex)*len(station_hex)*((end_time-start_time)//time_interval)，其每一个位置为以由每辆车旅行时间构成的numpy数组
    '''
    travel_time = [ [ [None] * ((end_time-start_time)//time_interval) for j in range(len(station_hex)) ] for i in range(len(station_hex))]
    # 创建len(station_hex)*len(station_hex)*((end_time-start_time)//time_interval)列表
    for index,row in data.iterrows(): 
        now_time = int(time.mktime(time.strptime(row['EXITTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
        if row['ENSTATIONHEX(varchar(20))'] in station_hex:
            o_station_index = station_hex.index(row['ENSTATIONHEX(varchar(20))'])
            d_station_index = station_hex.index(row['EXSTATIONHEX(varchar(20))'])
            if now_time >= start_time:
                entry_time = int(time.mktime(time.strptime(row['ENTRYTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
                one_traval_time = now_time-entry_time
                if travel_time[o_station_index][d_station_index][(now_time - start_time)//time_interval] is None:
                    travel_time[o_station_index][d_station_index][(now_time - start_time)//time_interval] = np.array(one_traval_time)
                else:
                    travel_time[o_station_index][d_station_index][(now_time - start_time)//time_interval] = np.append(travel_time[o_station_index][d_station_index][(now_time - start_time)//time_interval],one_traval_time)
    return travel_time

def get_travel_time(data , station_hex):  #需传入ex读出的data
    '''
    返回不随时间区间划分的旅行时间矩阵，shape为len(station_hex)*len(station_hex)*3，其中3个特征分别是平均时间，标准差，流量
    '''
    travel_time = [[None] * len(station_hex) for i in range(len(station_hex))]
    # 创建len(station_hex)*len(station_hex)*((end_time-start_time)//time_interval)列表
    for index,row in data.iterrows(): 
        if row['ENSTATIONHEX(varchar(20))'] in station_hex:
            o_station_index = station_hex.index(row['ENSTATIONHEX(varchar(20))'])
            d_station_index = station_hex.index(row['EXSTATIONHEX(varchar(20))'])
            now_time = int(time.mktime(time.strptime(row['EXITTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
            entry_time = int(time.mktime(time.strptime(row['ENTRYTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
            one_traval_time = now_time-entry_time
            if travel_time[o_station_index][d_station_index] is None:
                travel_time[o_station_index][d_station_index] = np.array(one_traval_time)
            else:
                travel_time[o_station_index][d_station_index] = np.append(travel_time[o_station_index][d_station_index],one_traval_time)
    for i in range(len(station_hex)):
        for j in range(len(station_hex)):
            if i==j:
                travel_time[i][j] = [0,0,0] 
            else:
                #3sita剔除异常值
                mean = np.mean(travel_time[i][j] , axis=0)
                std = np.std(travel_time[i][j] , axis=0)
                preprocessed_data_array = [x for x in travel_time[i][j] if (x > mean - 3 * std)]
                preprocessed_data_array = [x for x in preprocessed_data_array if (x < mean + 3 * std)]
                #加入剔除后的数组的均值，标准差，数量（流量）
                preprocessed_data_array = np.asarray(preprocessed_data_array)
                travel_time[i][j] = [np.mean(preprocessed_data_array , axis=0),np.std(preprocessed_data_array , axis=0),len(preprocessed_data_array)]
    return travel_time
def get_ex_station_lane_flow(data,station_hex):
    station_lane_flow = {station_hex[0]:0,station_hex[1]:0,station_hex[2]:0,station_hex[3]:0,station_hex[4]:0}
    #计数：每个站点对应的车道
    # count_list = [[],[],[],[],[]]
    for i in range(len(station_lane_flow)):
        # count_list[i] = data[data['EXSTATIONHEX(varchar(20))']==station_hex[i]].EXITLANE(int(11)).nunique() 
        station_lane_flow[station_hex[i]] = [ [0] * ((end_time-start_time)//time_interval) for j in range(8)]
    #编辑一个字典 让ENTRYLANE(int(11))能索引到0，1，2，3
    tempo_dic = {101:0,102:1,103:2,104:3,105:4,106:5,180:6,181:7}
    for index,row in data.iterrows(): 
        now_time = int(time.mktime(time.strptime(row['EXITTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
        station_lane_flow[row['EXSTATIONHEX(varchar(20))']][tempo_dic[row['EXITLANE(int(11))']]][(now_time - start_time)//time_interval] += 1
    for i in range(5):
        tempo_list = []
        for j in range(6):
            if np.array(station_lane_flow[station_hex[i]][j]).sum() > 300:
                tempo_list.append(station_lane_flow[station_hex[i]][j])
        station_lane_flow[station_hex[i]] = tempo_list.copy()
    
    return station_lane_flow
def get_en_station_lane_flow(data,station_hex):
    station_lane_flow = {station_hex[0]:0,station_hex[1]:0,station_hex[2]:0,station_hex[3]:0,station_hex[4]:0}
    #计数：每个站点对应的车道
    for i in range (len(station_lane_flow)):
        # station_lane_flow[station_hex[i]] = [ [0] * ((end_time-start_time)//time_interval) for j in range(6)]
        station_lane_flow[station_hex[i]] = [ [0] * ((end_time-start_time)//time_interval) for j in range(6)]
        #三个小时
    #编辑一个字典 让ENTRYLANE(int(11))能索引到0，1，2，3
    tempo_dic = {1:0,2:1,3:2,4:3,5:4,1001:5}
    for index,row in data.iterrows(): 
        now_time = int(time.mktime(time.strptime(row['ENTRYTIME(datetime)'][:-2], "%Y-%m-%d %H:%M:%S")))
        station_lane_flow[row['EXSTATIONHEX(varchar(20))']][tempo_dic[row['ENTRYLANE(int(11))']]][(now_time - start_time)//time_interval] += 1

    for i in range(5):
        tempo_list = []
        for j in range(6):
            if np.array(station_lane_flow[station_hex[i]][j]).sum() > 300:
                tempo_list.append(station_lane_flow[station_hex[i]][j])
        station_lane_flow[station_hex[i]] = tempo_list.copy()
    return station_lane_flow
def plot(ex_station_flow, station_hex,name):
    # x = [ i for i in range(0,(end_time-start_time)//time_interval)]
    # plt.plot(x,dict_ex_station_flow[hexnumber])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3,figsize=[16,9])
    
    fig.suptitle(name+'流量图', fontsize=20, fontweight="bold")
    fig.subplots_adjust(wspace=0.2,hspace=0.35)
    for i in range(6):
        if i < 3:
            axes[0][i].xaxis.set_major_locator(mdates.DayLocator(interval=5))
            axes[0][i].xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
            axes[0][i].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            axes[0][i].tick_params(axis = "both", direction = "out", labelsize = 10)
            axes[0][i].set_xlabel('time',fontsize=16)
            axes[0][i].set_ylabel('flow',fontsize=16)

        else:
            axes[1][i-3].xaxis.set_major_locator(mdates.DayLocator(interval=5))
            axes[1][i-3].xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
            axes[1][i-3].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            axes[1][i-3].tick_params(axis = "both", direction = "out", labelsize = 10)
            axes[1][i-3].set_xlabel('time',fontsize=16)
            axes[1][i-3].set_ylabel('flow',fontsize=16)
    dates = mdates.drange(date_start, data_end, delta)
    # x = [ i for i in range(0,len(ex_station_flow[0]))]
    for i in range(6):
        if i == 0:
            colors = ['r','b','g','y','c']
            for j in range(5):
                axes[0][i].plot(dates,ex_station_flow[j], color = colors[j],linewidth=0.5, linestyle='-',alpha = 0.7,label=station_hex[j]) 
            axes[0][i].set_title('汇总流量',y=-0.22,pad=-14,fontsize=18)
            axes[0][i].legend(loc='upper right',fontsize=16)
        elif i < 3:        
            axes[0][i].plot(dates,ex_station_flow[i-1], color = 'blue', linewidth=0.5,linestyle='-') 
            axes[0][i].set_title(station_hex[i-1],y=-0.24,pad=-14,fontsize=18)
        else:       
            axes[1][i-3].plot(dates,ex_station_flow[i-1], color = 'blue', linewidth=0.5,linestyle='-')
            axes[1][i-3].set_title(station_hex[i-1],y=-0.24,pad=-14,fontsize=18)
    return fig
def polt_heat(en_station_flow,en_station_lane_flow,station_hex,name):
    fig = plt.figure(figsize=[16,9])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.suptitle(name+'时空流量热力图', fontsize=20, fontweight="bold")
    plt.subplots_adjust(wspace=0.2,hspace=0.35)
    for i in range(6):
        if i == 0:
            plt.subplot(2, 3, i+1)
            sns.heatmap(en_station_flow,cmap = "mako_r")
            plt.title('站点级流量图',y=-0.24,pad=-14,fontsize=18)
            plt.xlabel('time',fontsize=18)
            plt.ylabel('station_num',fontsize=18)
        else:
            plt.subplot(2, 3, i+1)
            sns.heatmap(en_station_lane_flow[station_hex[i-1]],cmap = "mako_r")
            plt.title(station_hex[i-1]+'_车道级流量图',y=-0.24,pad=-14,fontsize=18)
            plt.xlabel('time',fontsize=18)
            plt.ylabel('lane_num',fontsize=18)
    return fig
def polt_od(travel_time):
    fig = plt.figure(figsize=[16,9])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.suptitle('OD热力图', fontsize=20, fontweight="bold")
    for i in range(3):
        if i==0:
            plt.subplot(2, 3, i+1)
            sns.heatmap(travel_time[:,:,0],cmap = "mako_r")
            plt.title('OD行程时间平均值热力图')
            plt.xlabel('station_num',fontsize=18)
            plt.ylabel('station_num',fontsize=18)
        elif i==1:
            plt.subplot(2, 3, i+1)
            sns.heatmap(travel_time[:,:,2],cmap = "mako_r")
            plt.title('OD流量热力图')
            plt.xlabel('station_num',fontsize=18)
            plt.ylabel('station_num',fontsize=18)
        elif i==2:
            plt.subplot(2, 3, i+1)
            sns.heatmap(travel_time[:,:,1],vmax=1200,cmap = "mako_r")
            plt.title('OD行程时间标准差热力图')
            plt.xlabel('station_num',fontsize=18)
            plt.ylabel('station_num',fontsize=18)
    return fig
# In[]            
def correlation_plot(en_flow,ex_flow):
    en_flow_2 = en_flow.transpose()
    ex_flow_2 = ex_flow.transpose()

    col = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    df_en_flow = pd.DataFrame(en_flow_2, columns=col)
    fig = plt.figure(figsize=[16,9])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    
    
    plt.subplot(1,2, 1)
    plt.title('入口相关性', fontsize=20, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14,rotation=90)
    sns.heatmap(df_en_flow.corr(),annot=True,annot_kws={"fontsize":20},cmap = "mako_r")
    
    
    col = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    df_ex_flow = pd.DataFrame(ex_flow_2, columns=col)
    
    
    plt.subplot(1,2, 2)
    plt.title('出口相关性', fontsize=20, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14,rotation=90)
    sns.heatmap(df_ex_flow.corr(),annot=True,annot_kws={"fontsize":20},cmap = "mako_r")
    return fig
# In[]  GUI接口
def process_entrance_data(path_en):
    station = ['2110002.0','2110003.0','2110004.0','2110005.0','2110006.0'] #2,3,4,5,6
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    dict_en_station_flow,en_data = get_en_station_flow(path_en,station,station_hex)
    en_station_flow = np.asarray(list(dict_en_station_flow.values()))
    en_station_lane_flow = get_en_station_lane_flow(en_data,station_hex)
    return en_station_flow, en_station_lane_flow

def process_exit_data(path_ex):
    # 读取和处理出口数据
    station = ['2110002.0','2110003.0','2110004.0','2110005.0','2110006.0'] #2,3,4,5,6
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    dict_ex_station_flow,ex_data = get_ex_station_flow(path_ex,station_hex)
    travel_time = get_travel_time(ex_data , station_hex)
    ex_station_flow = np.asarray(list(dict_ex_station_flow.values()))
    travel_time = np.asarray(travel_time)
    ex_station_lane_flow = get_ex_station_lane_flow(ex_data,station_hex)
    return ex_station_flow, travel_time, ex_station_lane_flow


# 主处理函数
def process_data(entrance_path, exit_path, output_dir):
    print("start process data")
    en_station_flow, en_station_lane_flow = process_entrance_data(entrance_path)
    print("process en data successfully")
    ex_station_flow,travel_time, ex_station_lane_flow = process_exit_data(exit_path)
    print("process ex data successfully")
    # 在保存文件之前创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 假设处理数据后保存为 .npy 文件
    np.save(f"{output_dir}/AENTRY.npy", en_station_flow)
    np.save(f"{output_dir}/AEXIT.npy", ex_station_flow)
    np.save(f"{output_dir}/AENTRY_lane.npy", en_station_lane_flow)
    np.save(f"{output_dir}/AEXIT_lane.npy", ex_station_lane_flow)
    np.save(f"{output_dir}/Travel_time.npy", travel_time)

def load_dict_from_numpy(filename): #读取字典
    return np.load(filename, allow_pickle=True).item()



def visualize(input_dir, figures_dict):
    station = ['2110002.0','2110003.0','2110004.0','2110005.0','2110006.0'] #2,3,4,5,6
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    
    en_station_flow = np.load(f"{input_dir}/AENTRY.npy")
    ex_station_flow = np.load(f"{input_dir}/AEXIT.npy")
    en_station_lane_flow = load_dict_from_numpy(f"{input_dir}/AENTRY_lane.npy")
    ex_station_lane_flow = load_dict_from_numpy(f"{input_dir}/AEXIT_lane.npy")
    travel_time = np.load(f"{input_dir}/Travel_time.npy")
    print("data_load_successfully")
    #entrance_path & time
    figures_dict[('entrance', 'time')] = plot(en_station_flow, station_hex,name = '入口站')
    #entrance_path & space
    figures_dict[('entrance', 'space')] = polt_heat(en_station_flow,en_station_lane_flow,station_hex,name = '入口站')
    #entrance_path & OD
    figures_dict[('entrance', 'OD feature')] = polt_od(travel_time)
    #entrance_path & correlation
    figures_dict[('entrance', 'correlation')] = correlation_plot(en_station_flow, ex_station_flow)
    print("entrance_fig_draw_successfully")
    
    #exit_path & time
    figures_dict[('exit', 'time')] = plot(ex_station_flow, station_hex,name = '出口站')
    #exit_path & space
    figures_dict[('exit', 'space')] = polt_heat(ex_station_flow,ex_station_lane_flow,station_hex,name = '出口站')
    #exit_path & OD
    figures_dict[('exit', 'OD feature')] = polt_od(travel_time)
    #exit_path & correlation    
    figures_dict[('exit', 'correlation')] = correlation_plot(en_station_flow, ex_station_flow)
    print("exit_fig_draw_successfully")
    return figures_dict
# In[]

if __name__ == '__main__':
    
# In[]
    path_ex = r'F:\AEXIT'
    # path_ex = r'F:\trial_for_biye\data_one_day_aexit'
    station = ['2110002.0','2110003.0','2110004.0','2110005.0','2110006.0'] #2,3,4,5,6
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    
    
    dict_ex_station_flow,ex_data = get_ex_station_flow(path_ex,station_hex)
    travel_time = get_travel_time(ex_data , station_hex)
    ex_station_flow = np.asarray(list(dict_ex_station_flow.values()))
    travel_time = np.asarray(travel_time)
    ex_station_lane_flow= get_ex_station_lane_flow(ex_data,station_hex)
# In[]    
    # travel_time = np.load('Travel_time.npy')
    polt_od(travel_time)
    
    plot(ex_station_flow, station_hex,name = '出口站')
    polt_heat(ex_station_flow,ex_station_lane_flow,station_hex,name = '出口站')
    polt_od(travel_time)
# In[]
    #读AENTRY
    path_en = r'F:\AENTRY'
    # path = r'E:\study\trial_for_biye\data_one_day_aentry'
    station = ['2110002.0','2110003.0','2110004.0','2110005.0','2110006.0'] #2,3,4,5,6
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    dict_en_station_flow,en_data = get_en_station_flow(path_en,station,station_hex)
# In[]
    en_station_flow = np.asarray(list(dict_en_station_flow.values()))
    en_station_lane_flow = get_en_station_lane_flow(en_data,station_hex)
    plot(en_station_flow, station_hex,name = '入口站')
    polt_heat(en_station_flow,en_station_lane_flow,station_hex,name = '入口站')
# In[]
    # # np.save('AEXIT.npy',ex_station_flow)
    # # np.save('Travel_time.npy',travel_time)
    # np.save('AENTRY.npy',en_station_flow)

# In[]
