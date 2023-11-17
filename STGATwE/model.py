# -*- coding: utf-8 -*-


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result

class TemporalConvLayer(nn.Module):

    

    def __init__(self, Kt, c_in, c_out, n_vertex):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.sigmoid = nn.Sigmoid()
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        
    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), self.sigmoid(x_q))
    
      
        return x
    
class STConvBlock(nn.Module):
    def __init__(self, Kt,  n_vertex, c_in, TemConv_out,GCN_out,edge_index,edge_attr, droprate):
        super(STConvBlock, self).__init__()
        self.conv1 = TemporalConvLayer(Kt, c_in, TemConv_out,n_vertex)
        self.conv2 = GATv2Conv(TemConv_out,GCN_out,edge_dim=3,dropout=0.1) #看下 是要改的
        # self.conv2 = GATv2Conv(TemConv_out,GCN_out,edge_dim=3,dropout=0.2)
        self.conv3 = TemporalConvLayer(Kt, GCN_out, TemConv_out, n_vertex)
        self.tc2_ln = nn.LayerNorm([n_vertex, TemConv_out])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    def forward(self, x):
        x = self.conv1(x)  #(batch_size, feature, time_step, n_vertex)
        x = x.permute(0, 3, 1, 2) #(batch_size,n_vertex,feature,time_step)
        outputs_x = []
        for i in range(x.shape[3]) :
            graph_signal = x[:, :, :, i] #(batch_size,n_vertex,feature)
            outputs_x_one = []
            for j in range(graph_signal.shape[0]):
                one_graph_signal = graph_signal[j,:,:] #(n_vertex,feature)
                output_x = self.conv2(one_graph_signal,edge_index = self.edge_index,edge_attr = self.edge_attr)
                outputs_x_one.append(torch.unsqueeze(output_x,dim=0)) #append(1,n_vertex,feature)
                
            outputs_x_one = torch.cat(outputs_x_one, dim=0) #append(batch_size,n_vertex,feature)
            outputs_x.append(outputs_x_one.unsqueeze(-1)) #append(batch_size,n_vertex,feature,-1)
        x = torch.cat(outputs_x, dim=-1) #(batch_size,n_vertex,feature,time_step)
        x = x.permute(0, 2, 3, 1) #(batch_size, feature, time_step, n_vertex)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #(batch_size, time_step, n_vertex, feature)->#(batch_size, feature, time_step, n_vertex)
        x = self.dropout(x)

        return x
    
class sim_STConvBlock(nn.Module):
    def __init__(self, Kt,  n_vertex, c_in, TemConv_out,GCN_out,edge_index,edge_attr, droprate):
        super(sim_STConvBlock, self).__init__()
        self.conv1 = TemporalConvLayer(Kt, c_in, TemConv_out,n_vertex)
        self.conv2 = GATv2Conv(TemConv_out,GCN_out,edge_dim=3,dropout=0.1) #看下 是要改的
        # self.conv2 = GATv2Conv(TemConv_out,GCN_out,edge_dim=3,dropout=0.2)
        # self.conv3 = TemporalConvLayer(Kt, GCN_out, TemConv_out, n_vertex)
        self.tc2_ln = nn.LayerNorm([n_vertex, GCN_out])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    def forward(self, x):
        x = self.conv1(x)  #(batch_size, feature, time_step, n_vertex)
        x = x.permute(0, 3, 1, 2) #(batch_size,n_vertex,feature,time_step)
        outputs_x = []
        for i in range(x.shape[3]) :
            graph_signal = x[:, :, :, i] #(batch_size,n_vertex,feature)
            outputs_x_one = []
            for j in range(graph_signal.shape[0]):
                one_graph_signal = graph_signal[j,:,:] #(n_vertex,feature)
                output_x = self.conv2(one_graph_signal,edge_index = self.edge_index,edge_attr = self.edge_attr)
                outputs_x_one.append(torch.unsqueeze(output_x,dim=0)) #append(1,n_vertex,feature)
                
            outputs_x_one = torch.cat(outputs_x_one, dim=0) #append(batch_size,n_vertex,feature)
            outputs_x.append(outputs_x_one.unsqueeze(-1)) #append(batch_size,n_vertex,feature,-1)
        x = torch.cat(outputs_x, dim=-1) #(batch_size,n_vertex,feature,time_step)
        x = x.permute(0, 2, 3, 1) #(batch_size, feature, time_step, n_vertex)
        x = self.relu(x)
        # x = self.conv3(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #(batch_size, time_step, n_vertex, feature)->#(batch_size, feature, time_step, n_vertex)
        x = self.dropout(x)

        return x

class STConvBlock_noedge(nn.Module):
    def __init__(self, Kt,  n_vertex, c_in, TemConv_out,GCN_out,edge_index, droprate):
        super(STConvBlock_noedge, self).__init__()
        self.conv1 = TemporalConvLayer(Kt, c_in, TemConv_out,n_vertex)
        self.conv2 = GATv2Conv(TemConv_out,GCN_out,dropout=0.1) #看下 是要改的
        # self.conv2 = GATv2Conv(TemConv_out,GCN_out,edge_dim=3,dropout=0.2)
        self.conv3 = TemporalConvLayer(Kt, GCN_out, TemConv_out, n_vertex)
        self.tc2_ln = nn.LayerNorm([n_vertex, TemConv_out])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.edge_index = edge_index
        
    def forward(self, x):
        x = self.conv1(x)  #(batch_size, feature, time_step, n_vertex)
        x = x.permute(0, 3, 1, 2) #(batch_size,n_vertex,feature,time_step)
        outputs_x = []
        for i in range(x.shape[3]) :
            graph_signal = x[:, :, :, i] #(batch_size,n_vertex,feature)
            outputs_x_one = []
            for j in range(graph_signal.shape[0]):
                one_graph_signal = graph_signal[j,:,:] #(n_vertex,feature)
                output_x = self.conv2(one_graph_signal,edge_index = self.edge_index)
                outputs_x_one.append(torch.unsqueeze(output_x,dim=0)) #append(1,n_vertex,feature)
                
            outputs_x_one = torch.cat(outputs_x_one, dim=0) #append(batch_size,n_vertex,feature)
            outputs_x.append(outputs_x_one.unsqueeze(-1)) #append(batch_size,n_vertex,feature,-1)
        x = torch.cat(outputs_x, dim=-1) #(batch_size,n_vertex,feature,time_step)
        x = x.permute(0, 2, 3, 1) #(batch_size, feature, time_step, n_vertex)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #(batch_size, time_step, n_vertex, feature)->#(batch_size, feature, time_step, n_vertex)
        x = self.dropout(x)

        return x

class STConvBlock_NOGAT(nn.Module):
    def __init__(self, Kt,  n_vertex, c_in, TemConv_out,GCN_out,edge_index, droprate):
        super(STConvBlock_NOGAT, self).__init__()
        self.conv1 = TemporalConvLayer(Kt, c_in, TemConv_out,n_vertex)
        self.conv2 = GCNConv(TemConv_out,GCN_out) #看下 是要改的
        self.conv3 = TemporalConvLayer(Kt, GCN_out, TemConv_out, n_vertex)
        self.tc2_ln = nn.LayerNorm([n_vertex, TemConv_out])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.edge_index = edge_index

    def forward(self, x):
        x = self.conv1(x)  #(batch_size, feature, time_step, n_vertex)
        x = x.permute(0, 3, 1, 2) #(batch_size,n_vertex,feature,time_step)
        outputs_x = []
        for i in range(x.shape[3]) :
            graph_signal = x[:, :, :, i] #(batch_size,n_vertex,feature)
            outputs_x_one = []
            for j in range(graph_signal.shape[0]):
                one_graph_signal = graph_signal[j,:,:] #(n_vertex,feature)
                output_x = self.conv2(one_graph_signal,edge_index = self.edge_index)
                outputs_x_one.append(torch.unsqueeze(output_x,dim=0)) #append(1,n_vertex,feature)
                
            outputs_x_one = torch.cat(outputs_x_one, dim=0) #append(batch_size,n_vertex,feature)
            outputs_x.append(outputs_x_one.unsqueeze(-1)) #append(batch_size,n_vertex,feature,-1)
        x = torch.cat(outputs_x, dim=-1) #(batch_size,n_vertex,feature,time_step)
        x = x.permute(0, 2, 3, 1) #(batch_size, feature, time_step, n_vertex)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #(batch_size, time_step, n_vertex, feature)->#(batch_size, feature, time_step, n_vertex)
        x = self.dropout(x)

        return x  

class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer
    #Ko 剩余的时间部
    def __init__(self, Ko, last_in, mid_out , end_out, n_vertex,  bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_in, mid_out, n_vertex)
        self.fc1 = nn.Linear(in_features=mid_out, out_features=mid_out, bias=bias)
        self.fc2 = nn.Linear(in_features=mid_out, out_features=end_out, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, mid_out])
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1)) #(batch_size, time_step, n_vertex, feature)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2) #->(batch_size, feature, time_step, n_vertex)
        return x
class STGCNGraphConv(nn.Module):      
    def __init__(self, Kt, n_vertex,edge_index,edge_attr):
        super(STGCNGraphConv, self).__init__()
        self.st1 = STConvBlock(Kt=Kt,n_vertex=n_vertex,c_in=2, TemConv_out=64,GCN_out=32,edge_index=edge_index, edge_attr =edge_attr , droprate=0.5)
        # self.st1 = STConvBlock(Kt=Kt,n_vertex=n_vertex,c_in=2, TemConv_out=64,GCN_out=32,edge_index=edge_index, edge_attr =edge_attr , droprate=0.5)
        self.st2 = STConvBlock(Kt=Kt,n_vertex=n_vertex,c_in=64, TemConv_out=64,GCN_out=32,edge_index=edge_index, edge_attr=edge_attr, droprate=0.5)
        Ko = 4  #delay-4-4
        self.output = OutputBlock(Ko=Ko, last_in=64, mid_out=128, end_out=2, n_vertex=n_vertex,bias=True, droprate=0.5)

    def forward(self, x):
        x = self.st1(x)
        x = self.st2(x)    
        x = self.output(x)
        return x
class STGCNGraphConv_noedge(nn.Module):      
    def __init__(self, Kt, n_vertex,edge_index):
        super(STGCNGraphConv_noedge, self).__init__()
        self.st1 = STConvBlock_noedge(Kt=Kt,n_vertex=n_vertex,c_in=2, TemConv_out=64,GCN_out=32,edge_index=edge_index , droprate=0.5)
        # self.st1 = STConvBlock(Kt=Kt,n_vertex=n_vertex,c_in=2, TemConv_out=64,GCN_out=32,edge_index=edge_index, edge_attr =edge_attr , droprate=0.5)
        self.st2 = STConvBlock_noedge(Kt=Kt,n_vertex=n_vertex,c_in=64, TemConv_out=64,GCN_out=32,edge_index=edge_index, droprate=0.5)
        Ko = 4  #delay-4-4
        self.output = OutputBlock(Ko=Ko, last_in=64, mid_out=128, end_out=2, n_vertex=n_vertex,bias=True, droprate=0.5)

    def forward(self, x):
        x = self.st1(x)
        x = self.st2(x)    
        x = self.output(x)
        return x

class sim_STGCNGraphConv(nn.Module):      
    def __init__(self, Kt, n_vertex,edge_index,edge_attr):
        super(sim_STGCNGraphConv, self).__init__()
        self.st1 = sim_STConvBlock(Kt=Kt,n_vertex=n_vertex,c_in=2, TemConv_out=64,GCN_out=32,edge_index=edge_index, edge_attr =edge_attr , droprate=0.5)
        # self.st1 = STConvBlock(Kt=Kt,n_vertex=n_vertex,c_in=2, TemConv_out=64,GCN_out=32,edge_index=edge_index, edge_attr =edge_attr , droprate=0.5)
        self.st2 = sim_STConvBlock(Kt=Kt,n_vertex=n_vertex,c_in=32, TemConv_out=64,GCN_out=32,edge_index=edge_index, edge_attr=edge_attr, droprate=0.5)
        Ko = 8  #delay-4-4
        self.output = OutputBlock(Ko=Ko, last_in=32, mid_out=128, end_out=2, n_vertex=n_vertex,bias=True, droprate=0.5)

    def forward(self, x):
        x = self.st1(x)
        x = self.st2(x)    
        x = self.output(x)
        return x

class STGCNGraphConv_NOGAT(nn.Module):      
    def __init__(self, Kt, n_vertex,edge_index):
        super(STGCNGraphConv_NOGAT, self).__init__()
        self.st1 = STConvBlock_NOGAT(Kt=Kt,n_vertex=n_vertex,c_in=2, TemConv_out=64,GCN_out=32,edge_index=edge_index , droprate=0.5)
        self.st2 = STConvBlock_NOGAT(Kt=Kt,n_vertex=n_vertex,c_in=64, TemConv_out=64,GCN_out=32,edge_index=edge_index, droprate=0.5)
        Ko = 4  #delay-4-4
        self.output = OutputBlock(Ko=Ko, last_in=64, mid_out=128, end_out=2, n_vertex=n_vertex,bias=True, droprate=0.5)
        
    
    def forward(self, x):
        x = self.st1(x)
        x = self.st2(x)    
        x = self.output(x)
        return x


class LSTM_GAT(nn.Module):
    def __init__(self,edge_index,edge_attr):
        super(LSTM_GAT, self).__init__()
        self.lstm1 = nn.LSTM(2,64,num_layers=1,dropout=0.2)
        self.GAT = GATConv(64,32)
        self.lstm2 = nn.LSTM(32,64,num_layers=1,dropout=0.2)
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,2)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    def forward(self,x):
        outputs = []
        for i in range(x.shape[3]):
            single = x[:,:,:,i].permute(2,0,1) #(batch_size, feature, time_step)->(time_step, batch_size,feature )
            single,hidden = self.lstm1(single) #(time_step, batch_size,feature)
            single = single.unsqueeze(-1) #(time_step,batch_size,feature,1)
            outputs.append(single) #append(time_step, batch_size,feature,1)
        x = torch.cat(outputs, dim=-1)  #(time_step, batch_size,feature,n_vertex)
        x = x.permute(1,3,2,0) #(batch_size, n_vertex,feature,time_step)
        outputs_x = []
        for i in range(x.shape[3]) :
            graph_signal = x[:, :, :, i] #(batch_size,n_vertex,feature)
            outputs_x_one = []
            for j in range(graph_signal.shape[0]):
                one_graph_signal = graph_signal[j,:,:] #(n_vertex,feature)
                output_x = self.GAT(one_graph_signal,edge_index = self.edge_index,edge_attr = self.edge_attr)
                outputs_x_one.append(torch.unsqueeze(output_x,dim=0)) #append(1,n_vertex,feature)
                
            outputs_x_one = torch.cat(outputs_x_one, dim=0) #append(batch_size,n_vertex,feature)
            outputs_x.append(outputs_x_one.unsqueeze(-1)) #append(batch_size,n_vertex,feature,-1)
        x = torch.cat(outputs_x, dim=-1) #(batch_size,n_vertex,feature,time_step)
        x = x.permute(0,2,3,1) #(batch_size, feature, time_step,n_vertex)
        outputs2 = []
        for i in range(x.shape[3]):
            single = x[:,:,:,i].permute(2,0,1) #(batch_size, feature, time_step)->(time_step, batch_size,feature )
            single,hidden = self.lstm2(single) #(time_step, batch_size,feature)
            single = single[-1].unsqueeze(-1) #(batch_size,feature,1)
            single = torch.unsqueeze(single,dim=0) #(1,batch_size,feature,1)
            outputs2.append(single) #append(time_step=1, batch_size,feature,1)
        x = torch.cat(outputs2, dim=-1)  #(time_step=1, batch_size,feature,n_vertex)
        x = F.relu(self.fc1(x.permute(1, 0, 3, 2))) #(batch_size, time_step, n_vertex, feature)
        x = self.fc2(x).permute(0, 3, 1, 2) #->(batch_size, feature, time_step, n_vertex)
        
        return x
        
        
class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = nn.Linear(2,128)  # 第一个隐含层  
        self.fc2 = nn.Linear(128,128)  # 第二个隐含层
        self.fc3 = nn.Linear(128,2)   # 输出层
        self.fc4 = nn.Linear(12,64)
        self.fc5= nn.Linear(64,1)
    def forward(self,x):
        # 前向传播， 输入值：x (batch_size, feature, time_step, n_vertex)
        x = F.relu(self.fc1(x.permute(0, 2, 3, 1)))  #(batch_size, time_step, n_vertex, feature)
        x = F.relu(self.fc2(x))   # 使用 relu 激活函数
        x = self.fc3(x).permute(0, 3, 2, 1)  #(batch_size, feature, n_vertex, time_step)
        x = F.relu(self.fc4(x))
        x = self.fc5(x).permute(0, 1, 3, 2) #(batch_size, feature, time_step, n_vertex)
        
        return x
    
class LSTMnet(torch.nn.Module):
    def __init__(self):
        super(LSTMnet,self).__init__()
        self.lstm = nn.LSTM(2,64,num_layers=2,dropout=0.2)
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,2)
    def forward(self,x):
        # 前向传播， 输入值：x (batch_size, feature, time_step, n_vertex)
        outputs = []
        for i in range(x.shape[3]):
            single = x[:,:,:,i].permute(2,0,1) #(batch_size, feature, time_step)->(time_step, batch_size,feature )
            single,hidden = self.lstm(single) #(time_step, batch_size,feature)
            single = single[-1].unsqueeze(-1) #(batch_size,feature,1)
            single = torch.unsqueeze(single,dim=0) #(1,batch_size,feature,1)
            outputs.append(single) #append(time_step=1, batch_size,feature,1)
        x = torch.cat(outputs, dim=-1)  #(time_step=1, batch_size,feature,n_vertex)
        x = F.relu(self.fc1(x.permute(1, 0, 3, 2))) #(batch_size, time_step, n_vertex, feature)
        x = self.fc2(x).permute(0, 3, 1, 2) #->(batch_size, feature, time_step, n_vertex)
        # x = F.softmax(x, dim=3).permute(0, 3, 1, 2) #->(batch_size, feature, time_step, n_vertex)
        
        return x


class GLUnet(torch.nn.Module):
    def __init__(self,Kt,n_vertex):
        super(GLUnet,self).__init__()
        self.glu1 = TemporalConvLayer(Kt, 2, 64, n_vertex)
        self.glu2 = TemporalConvLayer(Kt, 64, 64, n_vertex)
        Ko = 8  #delay-2-2
        self.output = OutputBlock(Ko=Ko, last_in=64, mid_out=128, end_out=2, n_vertex=n_vertex,bias=True, droprate=0.5)
    def forward(self,x):
        x = self.glu1(x)
        x = self.glu2(x)
        x = self.output(x)
        return x 
    


class gcn_net(torch.nn.Module):
    def __init__(self,edge_index):
        super(gcn_net,self).__init__()
        self.gcn = GCNConv(2,32)
        self.fc1 = nn.Linear(32,2)
        self.fc2 = nn.Linear(12,64)
        self.fc3= nn.Linear(64,1) 
        self.edge_index = edge_index
    def forward(self,x):
        # 前向传播， 输入值：x (batch_size, feature, time_step, n_vertex)
        x = x.permute(0, 3, 1, 2) #(batch_size,n_vertex,feature,time_step)
        outputs = []
        for i in range(x.shape[3]):
            single = x[:,:,:,i] #(batch_size,n_vertex,feature)
            single = self.gcn(single,edge_index=self.edge_index)  #(batch_size,n_vertex,feature)
            outputs.append(single.unsqueeze(-1)) #append(batch_size,n_vertex,feature,1)
        x = torch.cat(outputs, dim=-1) #(batch_size,n_vertex,feature,time_step)
        x = x.permute(0, 3, 1, 2) #(batch_size, time_step, n_vertex, feature)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x.permute(0,3,2,1))) #(batch_size, feature, n_vertex,time_step)
        x = self.fc3(x).permute(0,1,3,2)  #->(batch_size, feature, time_step, n_vertex)
        
        return x