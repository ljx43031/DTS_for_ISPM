#------------------------------------------------------------------------------
#运动模型估计网络MMEN
import torch
from torch import nn

class PartialInfoInfer_P1(nn.Module):
    def __init__(self, TIMESTEPS):
        super(PartialInfoInfer_P1, self).__init__()
        #网络结构设计：
        # 第一层：
        C1_cn = 4
        TIMESTEPS1 = TIMESTEPS - 3
        # CNN定义
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,               # input height
                out_channels=C1_cn,          # n_filters
                kernel_size=2,               # filter size
                stride=1,                    # filter movement/step
                padding=0),                  # if want same width and length of this image after con1d,
            )
        # 高斯参数定义
        self.Center1 = nn.Parameter(torch.ones(TIMESTEPS1, C1_cn).float())
        self.Sigma1 = nn.Parameter(torch.ones(TIMESTEPS1, C1_cn).float())
        
        # 第二层:
        C2_cn = 8
        TIMESTEPS2 = TIMESTEPS1 - C1_cn + 1
        # CNN定义
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,               # input height
                out_channels=C2_cn,          # n_filters
                kernel_size=C1_cn,           # filter size
                stride=1,                    # filter movement/step
                padding=0),                  # if want same width and length of this image after con1d,
            )
        # 高斯参数定义
        self.Center2 = nn.Parameter(torch.ones(TIMESTEPS2, C2_cn).float())
        self.Sigma2 = nn.Parameter(torch.ones(TIMESTEPS2, C2_cn).float())

        # 第三层:
        C3_cn = 16
        TIMESTEPS3 = TIMESTEPS2 - C2_cn + 1
        # CNN定义
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,               # input height
                out_channels=C3_cn,          # n_filters
                kernel_size=C2_cn,           # filter size
                stride=1,                    # filter movement/step
                padding=0),                  # if want same width and length of this image after con1d,
            )
        # 高斯参数定义
        self.Center3 = nn.Parameter(torch.ones(TIMESTEPS3, C3_cn).float())
        self.Sigma3 = nn.Parameter(torch.ones(TIMESTEPS3, C3_cn).float())
        
        # 第四层:
        StateTrans_num = 8
        # CNN定义--输出层
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,               # input height
                out_channels=StateTrans_num,          # n_filters
                kernel_size=C3_cn,           # filter size
                stride=1,                    # filter movement/step
                padding=0),                  # if want same width and length of this image after con1d,
            )
        
    def CDP1(self, x):
        # shape of x: BATCHSIZE * TIMESTEPS * INPUT_SIZE
        # 以时间为基准，从第一时刻的目标状态到其它时刻的状态做向量，计算向量（01）与其他向量（0t）的叉乘和点乘
        # 这个计算主要考虑向量夹角变化的信息，这个信息决定了坐标的转移矩阵。
        x_d0 = x - torch.unsqueeze(x[:, 0, :], 1)
        # 消除第一个点(时刻少1)
        x_d0 = x_d0[:, 1:, :]
        # Calculate the 2-Norm
        x_d0_2norm = torch.norm(x_d0, p=2, dim=2)
        TS = x_d0_2norm.size(1)
        # 计算第一个向量与其他向量之间的叉乘和点乘(时刻少1)
        # Calculate the corss and dot product
        x_d0_2norm_c = torch.unsqueeze(x_d0_2norm[:, 0], 1) * x_d0_2norm[:, 1:TS]
        x_d0_et0_0 = torch.unsqueeze(x_d0[:, 0, 0], 1)
        x_d0_et0_1 = torch.unsqueeze(x_d0[:, 0, 1], 1)
        # Cross product
        x_cross = (x_d0_et0_0 * x_d0[:, 1:TS, 1] - x_d0_et0_1 * x_d0[:, 1:TS, 0]) / x_d0_2norm_c
        # Dot product
        x_dot = (x_d0_et0_0 * x_d0[:, 1:TS, 0] + x_d0_et0_1 * x_d0[:, 1:TS, 1]) / x_d0_2norm_c
        # Combine cross and dot products
        x_comb = torch.stack([x_cross, x_dot], dim=2)
        return x_d0, x_comb
    
    def forward(self, x):
        # 数据信息统计
        # shape of x: BATCHSIZE * TIMESTEPS * INPUT_SIZE
        #计算目标状态序列基于第一个向量的累积差分以及叉乘和点乘的
        x_d0, x_cd = self.CDP1(x)
        #累积差分消除最后3个时刻值，确保其与数据转移信息对应
        x_d1 = x_d0[:, 0:x.shape[1]-3, :]
        #维度扩展出一维
        x_d1 = torch.unsqueeze(x_d1, 1)
        
        # 数据转移信息计算
        #通过卷积神经网络结合相邻向量的叉乘和点乘
        #第一层
        #维度扩展出一维
        x_cd = torch.unsqueeze(x_cd, 1)
        C1_mix = self.conv1(x_cd)
        C1_tran = torch.transpose(C1_mix, 1, 3)
        #计算节点信息      
        C1_i = torch.tanh(C1_tran)
        #计算节点状态
        C1_s = torch.exp(-(((C1_tran - self.Center1) / self.Sigma1)**2))
        #点乘结合
        C1_out = C1_i.mul(C1_s)

        #第二层
        C2_mix = self.conv2(C1_out)
        C2_tran = torch.transpose(C2_mix, 1, 3)
        #计算节点信息
        C2_i = torch.tanh(C2_tran)
        #计算节点状态
        C2_s = torch.exp(-(((C2_tran - self.Center2) / self.Sigma2)**2))
        #点乘结合
        C2_out = C2_i.mul(C2_s)

        #第三层
        C3_mix = self.conv3(C2_out)
        C3_tran = torch.transpose(C3_mix, 1, 3)  
        #计算节点信息
        C3_i = torch.tanh(C3_tran)
        #计算节点状态
        C3_s = torch.exp(-(((C3_tran - self.Center3) / self.Sigma3)**2))
        #点乘结合
        C3_out = C3_i.mul(C3_s)

        #输出层（状态转移矩阵信息）
        STM_i = self.conv4(C3_out)
        STM_i = torch.squeeze(STM_i)
        

        # 转移矩阵
        x_TM = STM_i.reshape(x.shape[0], 2, 4)
        x01 = torch.tensor([[1,0,0,0],[0,1,0,0]]).cuda().float()
        x01 = x01.reshape(1,2,4)
        x01_e = x01.repeat(x.shape[0], 1, 1)
        x_TM = torch.cat([x01_e, x_TM], 1)        
        return x_TM