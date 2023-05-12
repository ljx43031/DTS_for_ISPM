#------------------------------------------------------------------------------
#运动模型估计网络MMEN的训练
import numpy as np
import random as rd
import torch
from torch import nn
from scipy import signal as sn
import scipy.io as scio
from torch.autograd import Variable
from motion_model_estimation_network import PartialInfoInfer_P1
from noise_elimination_network import TLcell_sp
from Radar_Tracking_Environment import Trajectory_Generator_2D_av as TG2D

#------------------------------------------------------------------------------
#Parameter Setting-------------------------------------------------------------
#------------------------------------------------------------------------------
TIMESTEPS = 29
BATCHSIZE = 10
LR = 1e-3
#5个滤波器设计
b1, a1 = sn.butter(5, 0.8, btype='low')
b2, a2 = sn.butter(5, 0.6, btype='low')
b3, a3 = sn.butter(5, 0.4, btype='low')
b4, a4 = sn.butter(5, 0.2, btype='low')
b5, a5 = sn.butter(5, 0.05, btype='low')
#加载滤波网络
FNx = TLcell_sp(batchsize=BATCHSIZE)
FNx = FNx.cuda()
# FNx.load_state_dict(torch.load("D:/program/PartialInfo_Infer/Models/FN_LTC_sp_x_final_u2.pkl"))
FNx.load_state_dict(torch.load("D:/program/DT_ISPM_final/models/NEN_x_e_3.pkl"))
FNy = TLcell_sp(batchsize=BATCHSIZE)
FNy = FNy.cuda()
# FNy.load_state_dict(torch.load("D:/program/PartialInfo_Infer/Models/FN_LTC_sp_y_final_u2.pkl"))
FNy.load_state_dict(torch.load("D:/program/DT_ISPM_final/models/NEN_y_e_3.pkl"))

PII_P1 = PartialInfoInfer_P1(TIMESTEPS)
PII_P1 = PII_P1.cuda()
# 加载全部参数
PII_P1.load_state_dict(torch.load("D:/program/PartialInfo_Infer/Models/PII_P1_final_ft2.pkl"))
optimizer = torch.optim.Adam(PII_P1.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()
accu_st = 100

data_number = TIMESTEPS
#生成航迹,定义航迹类
Traj_Gen = TG2D(data_len = data_number, dis_n = 10, azi_n = 1)
train_N = 10000   #训练次数，一般取个较大的值，我们在训练中只看收敛情况。只要收敛到合适误差就停止训练
losses = []
for j in range(train_N):
    #初始化航迹
    #放一部分随机参数到Batch外面
    bp_distance = rd.uniform(1000, 10000)
    bp_dis_direction = (rd.random()-0.5) * 2 * 180
    bp_velocity = rd.uniform(50,350)
    bp_vel_direction = (rd.random()-0.5) * 2 * 180
    d_x = bp_distance * np.cos(bp_dis_direction*np.pi/180)  #The position of target along X dirction
    d_y = bp_distance * np.sin(bp_dis_direction*np.pi/180)  #The position of target along Y dirction
    v_x = bp_velocity * np.cos(bp_vel_direction*np.pi/180)  #The velocity of target along X dirction
    v_y = bp_velocity * np.sin(bp_vel_direction*np.pi/180)  #The velocity of target along Y dirction
    Traj_Gen.bp = np.array([[d_x, d_y, v_x, v_y]],'float64')

    my_traj_all = []
    my_obser_all = []
    my_F_all = []
    for i in range(BATCHSIZE):
        a_flag = np.random.randint(0,1)   #只建立2维匀速转弯的数据
        if a_flag == 0:
            Traj_Gen.turn_rate = rd.uniform(-90, 90)
        else:
            Traj_Gen.turn_rate = 0
            Traj_Gen.av = rd.uniform(0, 100)
            Traj_Gen.av_mode = np.random.randint(0,10)
            Traj_Gen.tau = rd.uniform(5, 50)
        my_traj, _, my_obser_n, my_obser_ns, my_F = Traj_Gen.trajectory()
        my_traj_all.append(my_traj)
        my_obser_all.append(my_obser_ns)                 #选择方位角经过smooth的数据
        my_F_all.append(my_F)
    
    my_trajs = np.asarray(my_traj_all)
    my_obsers = np.asarray(my_obser_all)
    my_Fs = np.asarray(my_F_all)
    
    #--------------------------------------------------------------------------
    #观测值的输入（三种方式：无噪输入，含噪输入，去噪输入）
    # #无噪输入
    # x_in = Variable(torch.from_numpy(my_trajs[:,:,0:2])).cuda().float()
    
    # #含噪输入
    # my_xin_x = np.cos(my_obsers[:,:,0])*my_obsers[:,:,1]
    # my_xin_y = np.sin(my_obsers[:,:,0])*my_obsers[:,:,1]
    # my_xin = np.stack([my_xin_x,my_xin_y],axis=2)
    # x_in = Variable(torch.from_numpy(my_xin)).cuda().float()
    
    #去噪输入
    my_xin_x = np.cos(my_obsers[:,:,0])*my_obsers[:,:,1]
    my_xin_y = np.sin(my_obsers[:,:,0])*my_obsers[:,:,1]
    #对x去噪
    my_xin = my_xin_x
    data_in_all = []
    for i in range(BATCHSIZE):
        my_xin_t = my_xin[i,:]
        data_in = [my_xin_t,]
        data_in.append(sn.filtfilt(b1, a1, my_xin_t))
        data_in.append(sn.filtfilt(b2, a2, my_xin_t))
        data_in.append(sn.filtfilt(b3, a3, my_xin_t))
        data_in.append(sn.filtfilt(b4, a4, my_xin_t))
        data_in.append(sn.filtfilt(b5, a5, my_xin_t))
        data_in = np.asarray(data_in)
        data_in = data_in.transpose(1,0)
        data_in_all.append(data_in)
    data_ins = np.asarray(data_in_all)
    #NEN去噪计算：
    data_ins = data_ins.transpose(1,0,2)   #timestep和batch互换
    data_ins = Variable(torch.from_numpy(data_ins)).cuda().float()
    data_out = FNx(data_ins)
    data_out = data_out[:,:,0]
    my_xin_x_en = data_out.transpose(1,0)
    #对y去噪
    my_xin = my_xin_y
    data_in_all = []
    for i in range(BATCHSIZE):
        my_xin_t = my_xin[i,:]
        data_in = [my_xin_t,]
        data_in.append(sn.filtfilt(b1, a1, my_xin_t))
        data_in.append(sn.filtfilt(b2, a2, my_xin_t))
        data_in.append(sn.filtfilt(b3, a3, my_xin_t))
        data_in.append(sn.filtfilt(b4, a4, my_xin_t))
        data_in.append(sn.filtfilt(b5, a5, my_xin_t))
        data_in = np.asarray(data_in)
        data_in = data_in.transpose(1,0)
        data_in_all.append(data_in)
    data_ins = np.asarray(data_in_all)
    #NEN去噪计算：
    data_ins = data_ins.transpose(1,0,2)   #timestep和batch互换
    data_ins = Variable(torch.from_numpy(data_ins)).cuda().float()
    data_out = FNy(data_ins)
    data_out = data_out[:,:,0]
    my_xin_y_en = data_out.transpose(1,0)
    x_in = torch.stack([my_xin_x_en, my_xin_y_en],dim=2)
 
    TM_p = PII_P1(x_in)
    real_Fs = Variable(torch.from_numpy(my_Fs)).cuda().float()
    
    optimizer.zero_grad()
    loss = loss_func(TM_p, real_Fs)
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    
    
    if j % accu_st == 0:
        my_loss = loss.cpu().data.numpy()
        losses.append(my_loss)
        print("step %d, The MSE of F prediction: %g" % ((j+1),my_loss))
        # torch.save(PII_P1.state_dict(), "D:/program/PartialInfo_Infer/Models/PII_P1_final.pkl")


# losses = np.asarray(losses)
# mydata = {'losses':losses}
# scio.savemat('MMEN_loss_WithoutNoise.mat', mydata)
