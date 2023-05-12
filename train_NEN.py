#------------------------------------------------------------------------------
#噪声消除网络NEN的训练
import numpy as np
import torch
from torch import nn
from scipy import signal as sn
import random as rd
import scipy.io as scio
from torch.autograd import Variable
from noise_elimination_network import TLcell_sp
from Radar_Tracking_Environment import Trajectory_Generator_2D_av as TG2D
#超参数定义
TIMESTEPS = 29
BATCHSIZE = 10
LR = 1e-3
WD = 0
#5个滤波器设计
b1, a1 = sn.butter(5, 0.8, btype='low')
b2, a2 = sn.butter(5, 0.6, btype='low')
b3, a3 = sn.butter(5, 0.4, btype='low')
b4, a4 = sn.butter(5, 0.2, btype='low')
b5, a5 = sn.butter(5, 0.05, btype='low')
data_number = TIMESTEPS
#生成航迹,定义航迹类
Traj_Gen = TG2D(data_len = data_number)

#网络模型实例化
FN = TLcell_sp(batchsize=BATCHSIZE)
FN = FN.cuda()
# FN.load_state_dict(torch.load("D:/program/PartialInfo_Infer/Models/FN_LTC_sp_x_final_u2.pkl"))
FN.load_state_dict(torch.load("D:/program/DT_ISPM_final/models/NEN_y_e_3.pkl"))
optimizer = torch.optim.Adam(FN.parameters(), lr=LR, weight_decay=WD)   # optimize all cnn parameters
loss_func = nn.MSELoss()
accu_st = 10

#训练开始：
#该网络只是负责一维数据的去噪工作。对于二维平面的雷达跟踪结果，我们有两个方向（也就是x,y方向上的含噪位置坐标序列）
#所以需要分别训练网络实现两个方向的去噪。
train_N = 10000   #训练次数，一般取个较大的值，我们在训练中只看收敛情况。只要收敛到合适误差就停止训练
losses = []
for j in range(train_N):
    #初始化航迹
    #放一部分随机参数到Batch外面
    bp_distance = rd.uniform(1000, 10000)
    bp_dis_direction = (rd.random()-0.5) * 2 * 180
    bp_velocity = rd.uniform(100,200)
    bp_vel_direction = (rd.random()-0.5) * 2 * 180
    
    d_x = bp_distance * np.cos(bp_dis_direction*np.pi/180)  #The position of target along X dirction
    d_y = bp_distance * np.sin(bp_dis_direction*np.pi/180)  #The position of target along Y dirction
    v_x = bp_velocity * np.cos(bp_vel_direction*np.pi/180)  #The velocity of target along X dirction
    v_y = bp_velocity * np.sin(bp_vel_direction*np.pi/180)  #The velocity of target along Y dirction
    Traj_Gen.bp = np.array([[d_x, d_y, v_x, v_y]],'float64')

    my_traj_all = []
    my_traj_n_all = []
    my_obser_all = []
    my_obser_n_all = []
    my_F_all = []
    data_in_all = []
    
    for i in range(BATCHSIZE):
        #转弯率设置：
        #----------------------------------------------------------------------
        #随机转弯率：
        a_flag = np.random.randint(0,1)   #只建立2维匀速转弯的数据,所以只产生0
        if a_flag == 0:
            Traj_Gen.turn_rate = rd.uniform(-90, 90)
        else:
            Traj_Gen.turn_rate = 0
            Traj_Gen.av = rd.uniform(0, 100)
            Traj_Gen.av_mode = np.random.randint(0,10)
            Traj_Gen.tau = rd.uniform(5, 50)
        #----------------------------------------------------------------------
        
        my_traj, my_traj_n, my_obser_n, my_obser_ns, my_F = Traj_Gen.trajectory()
        
        my_traj_all.append(my_traj)
        my_obser = my_obser_ns                                  #选择方位角经过smooth的数据
        #选择训练哪一维数据
        my_xin = np.cos(my_obser[:,0])*my_obser[:,1]           #模型参数：FN_LTC_sp_x   
        # my_xin = np.sin(my_obser[:,0])*my_obser[:,1]           #模型参数：FN_LTC_sp_y

        data_in = [my_xin,]
        data_in.append(sn.filtfilt(b1, a1, my_xin))
        data_in.append(sn.filtfilt(b2, a2, my_xin))
        data_in.append(sn.filtfilt(b3, a3, my_xin))
        data_in.append(sn.filtfilt(b4, a4, my_xin))
        data_in.append(sn.filtfilt(b5, a5, my_xin))
        data_in = np.asarray(data_in)
        data_in = data_in.transpose(1,0)
        data_in_all.append(data_in)
    
    my_trajs = np.asarray(my_traj_all)
    data_ins = np.asarray(data_in_all)
    
    #训练过程
    optimizer.zero_grad()
    
    #网络输入输出计算：
    x_test = data_ins.transpose(1,0,2)   #timestep和batch互换
    x_in = Variable(torch.from_numpy(x_test)).cuda().float()
    x_out = FN(x_in)
    
    #ground-truth构建：
    data_fs = data_ins[:,:,1:]
    data_ex = np.expand_dims(my_trajs[:,:,0], 2)                 #模型参数：FN_LTC_sp_x
    # data_ex = np.expand_dims(my_trajs[:,:,1], 2)                   #模型参数：FN_LTC_sp_y

    y_test = np.concatenate([data_ex, data_fs], 2)
    y_test = y_test.transpose(1,0,2)
    my_gt = Variable(torch.from_numpy(y_test)).cuda().float()
    
    #计算含频率信息的ground-truth时的loss
    loss = loss_func(x_out, my_gt)  

    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if j % accu_st == 0:
        my_loss = loss.cpu().data.numpy()
        losses.append(my_loss)
        print("step %d, The MSE of prediction: %g" % ((j+1),my_loss))
        # torch.save(FN.state_dict(), "D:\program\PartialInfo_Infer\Models/FN_LT.pkl")
        
# losses = np.asarray(losses)
# mydata = {'losses':losses}
# scio.savemat('NEN_x_loss.mat', mydata)

# #效果检测
# x_out = x_out.cpu().data.numpy()
# mydata = {'x_out':x_out, 'y_test':y_test}
# scio.savemat('test.mat', mydata)