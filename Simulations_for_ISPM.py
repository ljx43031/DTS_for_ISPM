#------------------------------------------------------------------------------
#最终的仿真
import scipy.io as scio
import numpy as np
import torch
from scipy import signal as sn
from torch.autograd import Variable
from DTS_maneuver_trajectory import Manuvering_trajectory
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import JulierSigmaPoints as SP
from motion_model_estimation_network import PartialInfoInfer_P1
from noise_elimination_network import TLcell_sp
import time

TIMESTEPS = 29
BATCHSIZE = 1
state_dim = 4     #目标状态向量维度
obser_dim = 2     #观测状态向量维度
sT = 0.1

#5个滤波器设计
b1, a1 = sn.butter(5, 0.8, btype='low')
b2, a2 = sn.butter(5, 0.6, btype='low')
b3, a3 = sn.butter(5, 0.4, btype='low')
b4, a4 = sn.butter(5, 0.2, btype='low')
b5, a5 = sn.butter(5, 0.05, btype='low')

#加载滤波网络
FNx = TLcell_sp(batchsize=BATCHSIZE)
FNx = FNx.cuda()
FNx.load_state_dict(torch.load("D:/program/DT_ISPM_final/models/NEN_x_e_3.pkl"))
FNy = TLcell_sp(batchsize=BATCHSIZE)
FNy = FNy.cuda()
FNy.load_state_dict(torch.load("D:/program/DT_ISPM_final/models/NEN_y_e_3.pkl"))
#加载运动模型估计网络
PII_P1 = PartialInfoInfer_P1(TIMESTEPS)
PII_P1 = PII_P1.cuda()
# 加载全部参数
PII_P1.load_state_dict(torch.load("D:/program/DT_ISPM_final/models/MMEN_final.pkl"))

#测试航迹的生成
#航迹噪声基本参数---------------------------------------------------------------
State_Noise_SD = 10     #目标状态噪声标准差
Distance_Noise_SD = 7  #观测距离噪声标准差
Azimuth_Noise_SD = 5    #观测方位角噪声标准差的1000倍

#------------------------------------------------------------------------------
#航迹1
d_x = -20000   #目标x方位
d_y = -5000   #目标y方位
v_x = 250     #目标x方向速度
v_y = 180    #目标x方向速度
TR = [-3, 8, 0]                 #每一段的转弯率
TS = [300, 300, 300]          #每一段转弯率对应的time steps

# #------------------------------------------------------------------------------
# #航迹2
# d_x = 12000   #目标x方位
# d_y = 13000   #目标y方位
# v_x = 250     #目标x方向速度
# v_y = -300    #目标x方向速度
# TR = [6, -30, 15, 2, 60, -5]                 #每一段的转弯率
# TS = [210, 90, 90, 210, 90, 210]          #每一段转弯率对应的time steps
#生产航迹
BP = np.array([[d_x, d_y, v_x, v_y]],'float64')
Traj_Gen, my_traj_all, my_obser_all, my_obser_n_all = Manuvering_trajectory(BP, State_Noise_SD, Distance_Noise_SD, Azimuth_Noise_SD, TR, TS)
    
data_number = np.sum(TS)

#跟踪初始化
#State transition function with constant velocity model
def my_fx(x, sT, TM):
    """ state transition function for sstate [downrange, vel, altitude]"""
    return np.dot(x, TM)

#Observation function
def my_hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r = np.array([0,0],'float64')
    r[0] = np.arctan2(x[1],x[0])
    r[1] = np.sqrt(np.square(x[0])+np.square(x[1]))
    return r

#定义UKF
pos_noise = 20
vel_noise = 2
my_SP = SP(state_dim,kappa=0.) 
my_UKF = UKF(dim_x=state_dim, dim_z=obser_dim, dt=sT, hx=my_hx, fx=my_fx, points=my_SP)
my_UKF.Q *= Traj_Gen.var_m
my_UKF.R *= Traj_Gen.R
x0_noise = np.array([np.random.normal(0,pos_noise,2),np.random.normal(0,vel_noise,2)])
x0 = my_traj_all[0,:] + np.reshape(x0_noise,[4,])
# x0 = my_traj_all[0,:]
my_UKF.x = x0
my_UKF.P *= 100.

##-----------------------------------------------------------------------------
#目标跟踪开始
TM_u = np.array([[1,0,0,0],[0,1,0,0],[sT,0,1,0],[0,sT,0,1]],'float64') #F_cv
xs = []
T1 = time.time()
for j in range(data_number):
    #截断29个时刻形成一个目标量测序列，用以输入到网络中预测目标转移矩阵
    if j <= data_number - 29:
        obser_t = my_obser_n_all[j:j+29,:]
        
        # #----------------------------------------------------------------------
        # #输入数据前期处理第一种方式：直接含噪输入
        # my_xin_x = np.cos(obser_t[:,0])*obser_t[:,1]
        # my_xin_y = np.sin(obser_t[:,0])*obser_t[:,1]  
        # my_xin_n = np.stack([my_xin_x,my_xin_y],axis=1)
        # x_in_xy = Variable(torch.from_numpy(my_xin_n)).cuda().float()
        # x_in_xy = x_in_xy.unsqueeze(0)
        
        #----------------------------------------------------------------------
        #含噪数据处理第二种方式：x y网络滤波
        my_xin = np.cos(obser_t[:,0])*obser_t[:,1]            #模型参数：FN_LTC_sp_x
        data_in = [my_xin,]
        data_in.append(sn.filtfilt(b1, a1, my_xin))
        data_in.append(sn.filtfilt(b2, a2, my_xin))
        data_in.append(sn.filtfilt(b3, a3, my_xin))
        data_in.append(sn.filtfilt(b4, a4, my_xin))
        data_in.append(sn.filtfilt(b5, a5, my_xin))
        data_in = np.asarray(data_in)
        data_in_x = data_in.transpose(1,0)
        my_xin = np.sin(obser_t[:,0])*obser_t[:,1]            #模型参数：FN_LTC_sp_y
        data_in = [my_xin,]
        data_in.append(sn.filtfilt(b1, a1, my_xin))
        data_in.append(sn.filtfilt(b2, a2, my_xin))
        data_in.append(sn.filtfilt(b3, a3, my_xin))
        data_in.append(sn.filtfilt(b4, a4, my_xin))
        data_in.append(sn.filtfilt(b5, a5, my_xin))
        data_in = np.asarray(data_in)
        data_in_y = data_in.transpose(1,0)
        #网络降噪：x y
        x_test = np.expand_dims(data_in_x, 1)
        x_in = Variable(torch.from_numpy(x_test)).cuda().float()
        filter_out_x = FNx(x_in)
        x_test = np.expand_dims(data_in_y, 1)
        x_in = Variable(torch.from_numpy(x_test)).cuda().float()
        filter_out_y = FNy(x_in)
        x_in_xy = torch.stack([filter_out_x[:,:,0], filter_out_y[:,:,0]],dim=2)
        x_in_xy = x_in_xy.transpose(1,0)   #timestep和batch互换
        
        #F预测
        TM_p = PII_P1(x_in_xy)
        Fp = TM_p.cpu().data.numpy()
        TM_u = np.squeeze(Fp)
    my_UKF.update(my_obser_all[j,:])   
    xs.append(my_UKF.x.copy())
    my_UKF.predict(fx_args=TM_u)
T2 = time.time()
    
# Traj_estimated = np.asarray(xs)
# mydata = {'my_traj':my_traj_all, 'traj_pred':Traj_estimated, 'my_obser_all':my_obser_all}
# scio.savemat('simulation_low_maneuvering_2.mat', mydata)