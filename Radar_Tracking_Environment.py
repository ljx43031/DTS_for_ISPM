# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:47:28 2020

@author: Administrator
"""

#Radar Tracking Environment

import numpy as np
#import torch
#from torch import nn
#import torch.nn.functional as F
#from torch.autograd import Variable
from scipy.linalg import cholesky as Matrix_sqrt
#from scipy.linalg import sqrtm as Matrix_sqrt
#from filterpy.kalman.tests.test_ukf import*

#2D Trajectories
class Trajectory_Generator_2D(object):
    def __init__(self, sT = 0.1, data_len = 50, state_n = 10, bp_distance = 1000, bp_dis_direction = 30,
                 bp_velocity = 100, bp_vel_direction = 30, dis_n = 10, azi_n = 8, TR = 0):
        
        #Sampling interval (s)
        self.sT = sT
        #Number of Sampling Points
        self.N = data_len
        
        #Beginning Point
        #bp_distance(m), bp_dis_direction(°), bp_velocity(m/s), bp_vel_direction(°)
        d_x = bp_distance * np.cos(bp_dis_direction*np.pi/180)   #Target X dirction position
        d_y = bp_distance * np.sin(bp_dis_direction*np.pi/180)   #Target Y dirction position
        v_x = bp_velocity * np.cos(bp_vel_direction*np.pi/180)  #Target X dirction velocity
        v_y = bp_velocity * np.sin(bp_vel_direction*np.pi/180)  #Target Y dirction velocity
        self.bp = np.array([[d_x, d_y, v_x, v_y]],'float64')
        
        #Transition noise（Acceleration noise m/s^2）
        s_var = np.square(state_n)
        T2 = np.power(sT,2)
        T3 = np.power(sT,3)
        T4 = np.power(sT,4)
        #var_m = np.array([[T4/4,0,T3/2,0],[0,T4/4,0,T3/2],[T3/2,0,T2,0],[0,T3/2,0,T2]]) * s_var
        self.var_m = np.array([[T4/4,0,0,0],[0,T4/4,0,0],[0,0,T2,0],[0,0,0,T2]]) * s_var
        self.chol_var = Matrix_sqrt(self.var_m)
        
        #observation noise (rad, m)
        self.azi_n = azi_n/1000
        self.dis_n = dis_n
        
        self.R = np.array([[(azi_n/1000)**2,0],[0,dis_n**2]])
        #turn rate
        self.turn_rate = TR
        
    def trajectory(self):
        
        if self.turn_rate == 0:
            F_c = np.array([[1,0,self.sT,0],[0,1,0,self.sT],[0,0,1,0],[0,0,0,1]])
        else:
            w = self.turn_rate*np.pi/180  #turn rate
            F_c = np.array([[1,0,np.sin(w*self.sT)/w,(np.cos(w*self.sT)-1)/w],
                            [0,1,-(np.cos(w*self.sT)-1)/w,np.sin(w*self.sT)/w],
                            [0,0,np.cos(w*self.sT),-np.sin(w*self.sT)],
                            [0,0,np.sin(w*self.sT),np.cos(w*self.sT)]])
        F_c = np.transpose(F_c,[1,0])
        Tj = np.array([[0 for i in range(4)] for j in range(self.N)],'float64') #Initialization of trajectory
        state_temp = self.bp
        for i in range(self.N):
            Tj[i,:] = state_temp
            state_temp = np.dot(state_temp, F_c)
        Tj_n = Tj + np.dot(np.random.randn(self.N, 4), self.chol_var)  #Add Noise
        
        Obser = np.array([[0 for i in range(2)] for j in range(self.N)],'float64') #Initialization of observation
#        #Observations without noise
#        Obser[:,0] = np.arctan2(Tj[:,1],Tj[:,0])  #Azimuth
#        Obser[:,1] = np.sqrt(np.square(Tj[:,0])+np.square(Tj[:,1]))  #Distance
        #Observations with noise
        Obser[:,0] = np.arctan2(Tj_n[:,1],Tj_n[:,0]) + np.random.normal(0,self.azi_n,self.N) #Azimuth
        Obser[:,1] = np.sqrt(np.square(Tj_n[:,0])+np.square(Tj_n[:,1])) + np.random.normal(0,self.dis_n,self.N)   #Distance
        
        return Tj, Tj_n, Obser, F_c

#2D Trajectories
class Trajectory_Generator_2D_av(object):
    def __init__(self, sT = 0.1, data_len = 50, state_n = 10, bp_distance = 1000, bp_dis_direction = 30,
                 bp_velocity = 100, bp_vel_direction = 30, dis_n = 10, azi_n = 8, TR = 0, av = 0, av_mode = 0, tau = 30):
        
        #Sampling interval (s)
        self.sT = sT
        #Number of Sampling Points
        self.N = data_len
        
        #Beginning Point
        #bp_distance(m), bp_dis_direction(°), bp_velocity(m/s), bp_vel_direction(°)
        d_x = bp_distance * np.cos(bp_dis_direction*np.pi/180)   #Target X dirction position
        d_y = bp_distance * np.sin(bp_dis_direction*np.pi/180)   #Target Y dirction position
        v_x = bp_velocity * np.cos(bp_vel_direction*np.pi/180)  #Target X dirction velocity
        v_y = bp_velocity * np.sin(bp_vel_direction*np.pi/180)  #Target Y dirction velocity
        self.bp = np.array([[d_x, d_y, v_x, v_y]],'float64')
        
        #Transition noise（Acceleration noise m/s^2）
        s_var = np.square(state_n)
        T2 = np.power(sT,2)
        T3 = np.power(sT,3)
        T4 = np.power(sT,4)
        #var_m = np.array([[T4/4,0,T3/2,0],[0,T4/4,0,T3/2],[T3/2,0,T2,0],[0,T3/2,0,T2]]) * s_var
        self.var_m = np.array([[T4/4,0,0,0],[0,T4/4,0,0],[0,0,T2,0],[0,0,0,T2]]) * s_var
        self.chol_var = Matrix_sqrt(self.var_m)
        
        #observation noise (rad, m)
        self.azi_n = azi_n/1000
        self.dis_n = dis_n
        
        self.R = np.array([[(azi_n/1000)**2,0],[0,dis_n**2]])
        #turn rate
        self.turn_rate = TR
        self.av = av
        self.av_mode = av_mode
        self.tau = tau
        self.alpha = 1/self.tau
        self.AT = self.alpha * self.sT
        self.E_AT = np.exp(- self.AT)
        self.AV_M0 = np.array([[(self.sT**2)/2, self.sT]],'float64')
        self.AV_M1 = np.array([[(self.AT - 1 + self.E_AT)/(self.alpha**2), (1 - self.E_AT)/self.alpha, self.E_AT]],'float64')

    def azimuth_smooth(self, x):
        xl = np.size(x)
        for i in range(xl-1):
            if x[i] - x[i+1] > np.pi:
                x[i+1] = x[i+1] + 2 * np.pi
            if x[i] - x[i+1] < -np.pi:
                x[i+1] = x[i+1] - 2 * np.pi
        return x
            
    def trajectory(self):
        
        if self.turn_rate == 0:
            F_c = np.array([[1,0,self.sT,0],[0,1,0,self.sT],[0,0,1,0],[0,0,0,1]])
        else:
            w = self.turn_rate*np.pi/180  #turn rate
            F_c = np.array([[1,0,np.sin(w*self.sT)/w,(np.cos(w*self.sT)-1)/w],
                            [0,1,-(np.cos(w*self.sT)-1)/w,np.sin(w*self.sT)/w],
                            [0,0,np.cos(w*self.sT),-np.sin(w*self.sT)],
                            [0,0,np.sin(w*self.sT),np.cos(w*self.sT)]])
        F_c = np.transpose(F_c,[1,0])
        Tj = np.array([[0 for i in range(4)] for j in range(self.N)],'float64') #Initialization of trajectory
        state_temp = self.bp
        for i in range(self.N):
            Tj[i,:] = state_temp
            if self.av == 0:
                state_temp = np.dot(state_temp, F_c)
            else:
                if self.av_mode == 0:
                    state_temp = np.dot(state_temp, F_c)
                    a_comp = self.av * self.AV_M0
                    state_temp = state_temp + np.array([a_comp[0,0], a_comp[0,0], a_comp[0,1], a_comp[0,1]])
                else:
                    state_temp = np.dot(state_temp, F_c)
                    a_comp = self.av * self.AV_M1
                    state_temp = state_temp + np.array([a_comp[0,0], a_comp[0,0], a_comp[0,1], a_comp[0,1]])
                    self.av = a_comp[0,2]
                    
        Tj_n = Tj + np.dot(np.random.randn(self.N, 4), self.chol_var)  #Add Noise
        #Observations without noise
        Obser = np.array([[0 for i in range(2)] for j in range(self.N)],'float64') #Initialization
        Obser[:,0] = self.azimuth_smooth(np.arctan2(Tj[:,1],Tj[:,0]))  #Azimuth
        # Obser[:,0] = np.arctan2(Tj[:,1],Tj[:,0])  #Azimuth without smooth
        Obser[:,1] = np.sqrt(np.square(Tj[:,0])+np.square(Tj[:,1]))  #Distance        

#这里有个问题是，当我smooth了方位角后，跟踪是有问题的，所以我保留smooth前的方位角  
        #Observations with noise
        Obser_n = np.array([[0 for i in range(2)] for j in range(self.N)],'float64') #Initialization of observation
        Obser_ns = np.array([[0 for i in range(2)] for j in range(self.N)],'float64') #Initialization of observation
        azimuth_n = np.random.normal(0,self.azi_n,self.N)
        Obser_n[:,0] = np.arctan2(Tj_n[:,1],Tj_n[:,0]) + azimuth_n #Azimuth
        Obser_ns[:,0] = self.azimuth_smooth(np.arctan2(Tj_n[:,1],Tj_n[:,0])) + azimuth_n #Azimuth
        # Obser_n[:,0] = np.arctan2(Tj_n[:,1],Tj_n[:,0]) + np.random.normal(0,self.azi_n,self.N)  #Azimuth without smooth
        Obser_n[:,1] = np.sqrt(np.square(Tj_n[:,0])+np.square(Tj_n[:,1])) + np.random.normal(0,self.dis_n,self.N)   #Distance
        Obser_ns[:,1] = Obser_n[:,1]
        return Tj, Tj_n, Obser_n, Obser_ns, F_c