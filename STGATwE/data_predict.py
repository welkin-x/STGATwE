# -*- coding: utf-8 -*-

import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd
import os
import json
import model as models
from torch_geometric.data import data as D
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0).tolist(), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
       
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data

def toplot(pre_data,data,kind):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3,figsize=(16,9))
    fig.suptitle(kind, fontsize=20, fontweight="bold")
    fig.subplots_adjust(wspace=0.2,hspace=0.35)
    plt.delaxes(axes[1, 2])
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    number = ['(a) ','(b) ','(c) ','(d) ','(e) ']
    x = [ i for i in range(0,len(pre_data))]
    for i in range(5):
        if i < 3:        
            axes[0][i].plot(x,data[:,i], color = 'blue', linestyle='-',label='true')
            axes[0][i].plot(x,pre_data[:,i],color = 'red', linestyle='--',label='pres')
            axes[0][i].legend(loc='upper right',fontsize=16)
            axes[0][i].set_xlabel('time',fontsize=16)
            axes[0][i].set_ylabel('flow',fontsize=16)
            axes[0][i].set_title(number[i]+station_hex[i]+'站流量预测结果',y=-0.22,pad=-14,fontsize=18)
        else:       
            axes[1][i-3].plot(x,data[:,i], color = 'blue', linestyle='-',label='true')
            axes[1][i-3].plot(x,pre_data[:,i],color = 'red', linestyle='--',label='pres')
            axes[1][i-3].legend(loc='upper right',fontsize=16)
            axes[1][i-3].set_xlabel('time',fontsize=16)
            axes[1][i-3].set_ylabel('flow',fontsize=16)
            axes[1][i-3].set_title(number[i]+station_hex[i]+'站流量预测结果',y=-0.22,pad=-14,fontsize=18)
    return fig
class ForecastDataset(torch_data.Dataset):
    def __init__(self, en_data,ex_data, window_size, delay,kind, normalize_method=None, norm_statistic=None, interval=1):
        #改horizon为delay，将范围改为未来delay时刻的某个点
        #传入en和ex的data
        self.window_size = window_size
        self.interval = interval
        self.delay = delay
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        self.en_data = en_data  
        self.ex_data = ex_data
        
        if normalize_method:
            self.en_data, en_norm_stat = normalized(self.en_data, normalize_method, norm_statistic)
            self.ex_data, ex_norm_stat = normalized(self.ex_data, normalize_method, norm_statistic)
            with open('./handle_data/'+kind+'en_norm_stat.json', 'w') as f:
                json.dump(en_norm_stat, f)
            with open('./handle_data/'+kind+'ex_norm_stat.json', 'w') as f:
                json.dump(ex_norm_stat, f)   
        
        data = np.vstack((self.en_data,self.ex_data)) #先en再ex
        data = data.reshape(2,len(en_data),len(en_data[0])) #（特征，时间，空间）
        self.data = data
        self.df_length = len(data[0]) #在时间维度上的长度
        self.x_end_idx = self.get_x_end_idx()
        

    def __getitem__(self, index): #当xx[xx]时即会调用此函数
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[:,lo: hi]
        target_data = self.data[:,hi + self.delay-1] #改为单点
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.delay + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

def train(train_loader, valid_loader,model_kind,edge_index,edge_attr,epoch_count = 35):
    if model_kind == 'STGAT':
        model = models.STGCNGraphConv(Kt=3, n_vertex=5,edge_index=edge_index,edge_attr =edge_attr).to(device)
    if model_kind == 'sim_STGAT':
        model = models.sim_STGCNGraphConv(Kt=3, n_vertex=5,edge_index=edge_index,edge_attr =edge_attr).to(device)
    if model_kind == 'STGAT_noedge':
        model = models.STGCNGraphConv_noedge(Kt=3, n_vertex=5,edge_index=edge_index).to(device)
    
    if model_kind == 'STGCN':
        model = models.STGCNGraphConv_NOGAT(Kt=3, n_vertex=5,edge_index=edge_index).to(device)
    if model_kind == 'GLU':
        model = models.GLUnet(Kt=3,n_vertex=5).to(device) 
    if model_kind =='MLP':
        model = models.MLP().to(device)
    if model_kind == 'LSTM':
        model = models.LSTMnet().to(device)
    if model_kind == 'GCN':
        model = models.gcn_net(edge_index=edge_index).to(device) 
    if model_kind == 'LSTM_GAT':
        model = models.LSTM_GAT(edge_index=edge_index,edge_attr =edge_attr).to(device)
    my_optim = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=0.5)
    forecast_loss = torch.nn.MSELoss(reduction='mean')
    fig,axe1=plt.subplots()
    axe1.set_title('训练')  #设置训练、验证的图
    axe1.set_xlabel('epoch')
    axe1.set_ylabel('loss')
    
    fig,axe2=plt.subplots()
    axe2.set_title('验证')
    axe2.set_xlabel('epoch')
    axe2.set_ylabel('loss')
    
    best_validate_mae = 100
    validate_score_non_decrease_count = 0
    for epoch in range(epoch_count):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            model.zero_grad()
            forecast = model(inputs)
            loss = forecast_loss(forecast[:,:,0,:], target)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        axe1.scatter(epoch,loss_total / cnt)  #散点

        if (epoch+1) % 5 == 0:  #调整学习率
            my_lr_scheduler.step()
        if (epoch + 1) % 5 == 0:  #验证
        # if epoch >=15:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            model.eval()
            loss_total_eval = 0
            cnt_val = 0
            for i, (inputs_val, target_val) in enumerate(valid_loader):
                inputs_val = inputs_val.to(device)
                target_val = target_val.to(device)
                with torch.no_grad():
                    pred = model(inputs_val)
                    loss_val = forecast_loss(pred[:,:,0,:], target_val)
                cnt_val += 1
                loss_total_eval += float(loss_val)
            axe2.scatter(epoch,loss_total_eval / cnt_val) #散点
            
            if best_validate_mae > loss_total_eval / cnt_val:
                best_validate_mae = loss_total_eval / cnt_val
                is_best_for_now = True
                torch.save(model,'./model/' + model_kind + '_model.pth')
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            print('valid_loss:',loss_total_eval / cnt_val,'best_validate_mae:',best_validate_mae,'validate_score_non_decrease_count:',validate_score_non_decrease_count)

def test(test_en_data,test_ex_data,test_loader,model_kind):
    en_result_list = [] #将其整合为(时间，节点)的格式
    ex_result_list = []
    model_test = torch.load('./model/' + model_kind + '_model.pth')
    
    model_test.eval()
    with open(os.path.join('./handle_data/'+'testen_norm_stat.json'),'r') as f:
        test_en_normalize_statistic = json.load(f)
    with open(os.path.join('./handle_data/'+'testex_norm_stat.json'),'r') as f:
        test_ex_normalize_statistic = json.load(f)

    for i, (inputs_test, target_test) in enumerate(test_loader):
        inputs_test = inputs_test.to(device)
        target_test = target_test.to(device)
        with torch.no_grad():
            result = model_test(inputs_test)
        result = np.array(result.cpu())
        en_result = result[0,0,0,:]
        ex_result = result[0,1,0,:]
        en_result_list.append(en_result)
        ex_result_list.append(ex_result)
    en_result_list=de_normalized(np.array(en_result_list), normalize_method='z_score', norm_statistic=test_en_normalize_statistic)
    ex_result_list=de_normalized(np.array(ex_result_list), normalize_method='z_score', norm_statistic=test_ex_normalize_statistic)            
    true_en_data = test_en_data[-len(en_result_list):]
    true_ex_data = test_ex_data[-len(ex_result_list):]
    sum_true_en_data = np.array(true_en_data).flatten()
    sum_true_ex_data = np.array(true_ex_data).flatten()
    sum_en_result_list = np.array(en_result_list).flatten()
    sum_ex_result_list = np.array(ex_result_list).flatten()
    for i in range(5):
        mae, rmse,r_2 = evaluation(true_en_data[:,i],en_result_list[:,i])
        print('入口站_',station_hex[i],':mae=',mae,',rmse=',rmse,',r_2=',r_2)
    mae, rmse,r_2 = evaluation(sum_true_en_data,sum_en_result_list)
    print('全部入口站：',':mae=',mae,',rmse=',rmse,',r_2=',r_2)
    for i in range(5):
        mae, rmse,r_2 = evaluation(true_ex_data[:,i],ex_result_list[:,i])
        print('出口站_',station_hex[i],':mae=',mae,',rmse=',rmse,'r_2=',r_2)
    mae, rmse,r_2 = evaluation(sum_true_ex_data,sum_ex_result_list)
    print('全部出口站：',':mae=',mae,',rmse=',rmse,',r_2=',r_2) 
    
    fig_en = toplot(en_result_list,true_en_data,kind='收费站入口流量')
    fig_ex = toplot(ex_result_list,true_ex_data,kind='收费站出口流量')
    
    return fig_en,fig_ex, mae, rmse,r_2


def evaluation(y_test, y_predict):  #传入每个站点的 即一维数据
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    # mape=(abs(y_predict -y_test)/ y_test).mean()
    r_2=r2_score(y_test, y_predict)
    return mae, rmse,r_2


def GUI_predict(input_dir, model_kind, if_train = False):
    
    en_flow = np.load(f"{input_dir}/AENTRY.npy")
    ex_flow = np.load(f"{input_dir}/AEXIT.npy")
    en_flow_2 = en_flow.transpose()
    ex_flow_2 = ex_flow.transpose()
    window_size = 12
    delay = 3
    train_index = (len(en_flow_2)//10) * 7
    valid_index = train_index + ((len(en_flow_2)//10) * 2)
    test_index = valid_index + ((len(en_flow_2)//10))
    station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
    
    train_en_data = en_flow_2[:train_index]
    train_ex_data = ex_flow_2[:train_index]
    valid_en_data = en_flow_2[train_index:valid_index]
    valid_ex_data = ex_flow_2[train_index:valid_index]
    test_en_data = en_flow_2[valid_index:-4*24]
    test_ex_data = ex_flow_2[valid_index:-4*24]
    edge_index = torch.tensor([[0,1,2,3,1,2,3,4],[1,2,3,4,0,1,2,3]],dtype = torch.long).to(device)
    '''
    标准化一下边的属性
    '''
    Travel_time_list = np.load('./data/Travel_time.npy')
    edge_attr_list = np.array([Travel_time_list[0,1],Travel_time_list[1,2],Travel_time_list[2,3],Travel_time_list[3,4],\
                              Travel_time_list[1,0],Travel_time_list[2,1],Travel_time_list[3,2],Travel_time_list[4,3]])
    edge_attr_list, edge_norm_stat = normalized(edge_attr_list, normalize_method='z_score')
    with open('./handle_data/'+'edge_norm_stat .json', 'w') as f:
        json.dump(edge_norm_stat , f)
    edge_attr = torch.tensor(edge_attr_list,dtype = torch.float).to(device)
    # edge_index.expand(32,2,8)
    # edge_attr.expand(32,8,3)

    if if_train == "YES":
        train_set = ForecastDataset(train_en_data,train_ex_data,window_size, delay,kind='train', normalize_method='z_score')
        valid_set = ForecastDataset(valid_en_data,valid_ex_data,window_size, delay,kind='valid', normalize_method='z_score')
        train_loader = torch_data.DataLoader(train_set, batch_size=32, drop_last=True, shuffle=True, num_workers=0)
        valid_loader = torch_data.DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=0)
        train(train_loader, valid_loader,model_kind,edge_index,edge_attr) 
    
    test_set = ForecastDataset(test_en_data,test_ex_data,window_size, delay,kind='test', normalize_method='z_score') 
    test_loader = torch_data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    fig_en,fig_ex, mae, rmse,r_2 = test(test_en_data,test_ex_data,test_loader,model_kind)
    return fig_en,fig_ex, mae, rmse,r_2
'''
参数
'''
device = torch.device('cuda')
station_hex = ['3201D302','3201D303','3201D304','3201D305','3201D306']
if __name__ == "__main__":
    GUI_predict(input_dir="./data", model_kind = 'STGAT', if_train = False)
    



