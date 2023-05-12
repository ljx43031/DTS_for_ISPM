#------------------------------------------------------------------------------
#基于数据孪生的机动航迹生成系统

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import JulierSigmaPoints as SP
from Radar_Tracking_Environment import Trajectory_Generator_2D_av as TG2D

#机动目标航迹生成
def Manuvering_trajectory(BP, State_Noise_SD, Distance_Noise_SD, Azimuth_Noise_SD, TR, TS):
    #输入：
    # BP:航迹起始点（论文中考虑二维平面，且没有加速度信息）[位置x,位置y,速度x,速度y]
    # State_Noise_SD:状态噪声，一般指状态噪声中的加速度噪声（距离和速度的噪声可以推导）,单位 m/s^2
    # Distance_Noise_SD:观测噪声——距离噪声，单位 m
    # Azimuth_Noise_SD:观测噪声——方位角噪声，单位 rad
    # TR: 一个列表，列表中每个值表示每一段的转弯率
    # TS：一个列表，维度跟TR一致，列表中每个值表示每一段的时间长度
    #输出：
    # Traj_Gen: 航迹生成单元，主要是需要里面的噪声信息
    # my_traj_all: 整段无噪航迹
    # my_obser_n_all: 有噪观测数值
    # my_obser_ns_all: 有噪观测数值,其中方位角经过平滑。所谓平滑，是指消除方位角从pi到-pi或者-pi到pi的跃变
    
    Traj_Gen = TG2D(state_n = State_Noise_SD, dis_n = Distance_Noise_SD, azi_n = Azimuth_Noise_SD)
    Traj_Gen.bp = BP
    my_traj_all = []
    my_obser_n_all = []
    my_obser_ns_all = []
    for tr, ts in zip(TR, TS):
        Traj_Gen.N = ts + 1
        Traj_Gen.turn_rate = tr
        my_traj, my_traj_n, my_obser_n, my_obser_ns, my_F = Traj_Gen.trajectory()
        my_traj_all.append(my_traj[:ts,:])
        my_obser_n_all.append(my_obser_n[:ts,:])
        my_obser_ns_all.append(my_obser_ns[:ts,:])
        Traj_Gen.bp = my_traj[-1,:]
    my_traj_all = np.vstack(my_traj_all)
    my_obser_n_all = np.vstack(my_obser_n_all)
    my_obser_ns_all = np.vstack(my_obser_ns_all)
    return Traj_Gen, my_traj_all, my_obser_n_all, my_obser_ns_all

if __name__ == '__main__':
    #测试与论文部分航迹数据生成
    #生成航迹
    
    # #------------------------------------------------------------------------------
    # #航迹1
    # #航迹噪声基本参数---------------------------------------------------------------
    # State_Noise_SD = 20     #目标状态噪声标准差
    # Distance_Noise_SD = 20  #观测距离噪声标准差
    # Azimuth_Noise_SD = 8    #观测方位角噪声标准差的1000倍
    # d_x = -20000   #目标x方位
    # d_y = -5000   #目标y方位
    # v_x = 250     #目标x方向速度
    # v_y = 180    #目标x方向速度
    # TR = [-3, 8, 0]                 #每一段的转弯率
    # TS = [300, 300, 300]          #每一段转弯率对应的time steps

    #------------------------------------------------------------------------------
    #航迹2
    #航迹噪声基本参数---------------------------------------------------------------
    State_Noise_SD = 20     #目标状态噪声标准差
    Distance_Noise_SD = 40  #观测距离噪声标准差
    Azimuth_Noise_SD = 4    #观测方位角噪声标准差的1000倍
    d_x = 10000   #目标x方位
    d_y = 20000   #目标y方位
    v_x = 150     #目标x方向速度
    v_y = 90    #目标x方向速度
    TR = [10, -5, 6, -2, 8]                 #每一段的转弯率
    TS = [200, 200, 200, 200, 200]          #每一段转弯率对应的time steps
    
    BP = np.array([[d_x, d_y, v_x, v_y]],'float64')
    Traj_Gen, my_traj_all, my_obser_n_all, my_obser_ns_all = Manuvering_trajectory(BP, State_Noise_SD, Distance_Noise_SD, Azimuth_Noise_SD, TR, TS)
    import scipy.io as scio
    mydata = {'my_trajs':my_traj_all, 'my_obsers_n':my_obser_n_all, 'my_obsers_ns':my_obser_ns_all}
    scio.savemat('traj_info.mat', mydata)