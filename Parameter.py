"""用于设置整个工程的参数"""
# ！！！需要设置各个随机初值的量级

import numpy as np
np.random.seed(47)

'''
Parameter List:
    M              = 5                                    ; number of APs
    N              = 30                                   ; number of WDs
    T              = 0.1 s                                ; duration of each time slot
    B              = 1 * 10^6 Hz                          ; wireless bandwidth
    A[i]           = random~[1*10^4, 5*10^4] bit          ; arrived computation data

    mu[i]          = 0.51                                 ; energy conversion efficiency
    k[i]           = 10^(-28)                             ; energy efficiency coefficient of cpu
    phi[i]         = 1000 cycles/bit                      ; number of cpu cycles required to process one bit of data
    sigma[j]^2     = 10^(-9) W                            ; noise power at APs
    v[i]           = 1.1                                  ; wireless communication overhead
    eta            = 8.2 * 10^(-9) J                      ; energy consumption per cycle in MEC server

    f_max[i]       = 0.5 * 10^9 Hz                        ; maximum cpu frequency of WDs
    P_T_max[j]     = 4 W                                  ; maximum power of WPT
    P_max[i]       = 0.2 W                                ; maximum power of computation offloading
    B_max[i]       = 1 * 10^4 J                           ; battery capacity

    Randomly generate the locations of WDs and APs in a 5m*5m square
    theta_U        = 6.25 * 10^(-4)                       ; uplink gain coefficient
    dist[i][k]     = random                               ; distance between WD i and AP j
    h_random[i][j] = CN(0,1)                              ; complex normal distribution（直接用虚数的模）
    h_U[i][j]      = theta_U * dist[i][k]^(-3) * h_random ; uplink channel gain
    h_D[i][j]      = 2 * h_U[i][j]                        ; downlink channel gain
'''


class Parameter(object):
    def __init__(self, V=3, N=30, M=5):
        super().__init__()

        '''常量'''

        # 调度程序中的迭代次数
        self.MAX_EPISODES = 500
        
        self.scalar = 100000 # 原本以bit作为单位，但是为了让数据队列的长度和电池队列的长度尺度一致，改成以10kb作为单位
        self.B_scalar = 1000
        self.Q_scalar = 1

        self.T = 0.1  # 每个时帧的长度
        self.B = 100. * 10**3 / self.scalar  # 传输带宽         需要与残余电量B_remain进行区别

        self.N = N  # 无线终端数
        self.M = M  # AP数

        self.V = V * self.B_scalar / 20
        # V is a tunable parameter that controls the trade-off between the energy consumption and the queueing delay of computation data
        self.mu = np.array([0.51 for _ in range(self.N)])
        # mu[i] is the energy conversion efﬁciency of WD i
        self.k = np.array([2 * 10**(-27) for _ in range(self.N)])
        # k[i] is the energy efﬁciency coefﬁcient of the chip equipped with WD i
        self.phi = np.array([1000. * self.scalar for _ in range(self.N)])
        # phi[i] is the number of CPU cycles required to process one bit of computation data
        self.sigma = np.array([10**(-4.5) for _ in range(self.M)])
        # sigma[j]^2 is the noise power of AP j
        self.v = np.array([1.1 for _ in range(self.N)])
        # v[i] > 1 indicates the communication overhead induced by encryption and packet header
        self.eta = 4 * 10**(-10)
        # self.eta = 8.2 * 10**(-9) / 10.
        # eta is the energy consumption per CPU cycle of APs

        self.f_max = np.array([1 * 10**9 for _ in range(self.N)])
        # WD i 的最大CPU频率
        self.P_max = np.array([0.3 for _ in range(self.N)])
        # WD i 的最大发射能量
        self.P_T_max = np.array([4. for _ in range(self.M)])
        # AP j 的最大发射能量
        self.B_max = np.array([1. for _ in range(self.N)])
        # WD i 的最大电池容量

        self.tau_L = np.array([self.T for _ in range(self.N)])
        # WD 在本地运算的时间（一个时隙内）

        '''AP与WD'''
        # 需要调用本class的方法进行更新

        self.locWD = np.array([[0., 0.] for _ in range(self.N)])
        # locWD[i] 是 WD i 在图中的横纵坐标
        self.locAP = np.array([[0., 0.] for _ in range(self.M)])
        # locAP[j] 是 AP j 在图中的横纵坐标
        self.h_D = np.array([[(np.random.rand()+1.5)*0.01 for _ in range(self.M)] for _ in range(self.N)])
        # h_D[i][j] 是 WD i 与 AP j 之间的下行增益
        self.h_U = np.array([[(np.random.rand()+0.5)*0.01 for _ in range(self.M)] for _ in range(self.N)])
        # h_U[i][j] 是 WD i 与 AP j 之间的上行增益



        '''变量'''

        # 环境中更新
        self.D_L = np.array([0 for _ in range(self.N)])  # D_L[i] 是 WD i 在本地计算的比特大小
        self.D_O = np.array([0 for _ in range(self.N)])  # D_O[i] 是 WD i 上传到边缘去计算的比特大小
        self.Q = np.array([0 for _ in range(self.N)])  # Q[i] 是 WD i 现在缓存在队列中的比特大小
        self.A = np.array([0 for _ in range(self.N)])  # A[i] 是 WD i 上新的待计算比特大小

        self.B_remain = np.array([0. for _ in range(self.N)])   # B_[i] 是 WD i 上残余的电量      需要与带宽B进行区别
        self.B_minus = np.array([0. for _ in range(self.N)])    # B_minus[i] 是 WD i 上消耗掉的电量

        self.E_H = np.array([0. for _ in range(self.N)])   # E_H[i] is the energy harvested by WD i during the t-th time slot
        self.E_L = np.array([0. for _ in range(self.N)])   # E_L[i] is the energy consumption for local computation
        self.E_O = np.array([0. for _ in range(self.N)])   # E_O[i] is the energy consumption for wireless computation

        # 算法中更新
        self.c_T = np.array([0. for _ in range(self.M)])
        self.a_T = np.array([0 for _ in range(self.M)])     # 指示哪个AP进行WPT
        self.P_T = np.array([0. for _ in range(self.M)])     # AP的功耗
        self.tau_T = np.array([0. for _ in range(self.M)])   # AP进行发送的时间（WPT的时间）
        self.f = np.array([0. for _ in range(self.N)])       # WD的cpu频率
        self.tau = np.array([0. for _ in range(self.N)])     # WD进行发送的时间
        self.a = np.array([[0 for _ in range(self.M)] for _ in range(self.N)])  # 指示WD连接哪个AP
        self.P = np.array([0. for _ in range(self.N)])       # WP的功耗

    '''生成AP与WD的分布地图'''
    def generateMap(self):
        # Map Size: 5m * 5m
        # Generate APs
        for j in range(self.M):
            while 1:
                xj = np.random.randint(0, 50) / 10.
                yj = np.random.randint(0, 50) / 10.
                isExist = False
                for [x, y] in self.locAP:
                    if x == xj and y == yj:
                        isExist = True
                        break
                if not isExist:
                    self.locAP[j] = [xj, yj]
                    break
        # Generate WDs
        for i in range(self.N):
            while 1:
                xi = np.random.randint(0, 50) / 10.
                yi = np.random.randint(0, 50) / 10.
                isExist = False
                for [x, y] in self.locWD:
                    if x == xi and y == yi:
                        isExist = True
                        break
                for [x, y] in self.locAP:
                    if x == xi and y == yi:
                        isExist = True
                        break
                if not isExist:
                    self.locWD[i] = [xi, yi]
                    break

        self.calChannelGain()
        print('Success to generate map!\n')

    '''从文件中读取AP与WD的分布地图'''
    def loadMap(self, filename: str):
        npzfile = np.load('data/'+filename)
        self.locAP = npzfile['arr_0']
        self.locWD = npzfile['arr_1']
        self.calChannelGain()
        print('Success to load map!')

    '''将AP与WD的分布地图保存到文件'''
    def saveMap(self, filename: str):
        np.savez('data/'+filename, self.locAP, self.locWD)
        print('Success to save map!\n')

    '''将分布地图打印到文件中'''
    def printMap(self, filename: str):
        # A代表AP W代表WD
        # 以0.1m为间隔
        map = [[' ' for _ in range(50)] for _ in range(50)]
        for i in range(self.N):
            x = int(self.locWD[i][0] * 10)
            y = int(self.locWD[i][1] * 10)
            map[x][y] = 'W'
        for j in range(self.M):
            x = int(self.locAP[j][0] * 10)
            y = int(self.locAP[j][1] * 10)
            map[x][y] = 'A'
        np.savetxt('data/'+filename, map, fmt='%c')
        print('Success to print map!\n')

    '''计算信道的上下行增益'''
    def calChannelGain(self):
        theta_U = 6.25 * 10**(-4)
        for i in range(self.N):
            xi, yi = self.locWD[i]
            for j in range(self.M):
                xj, yj = self.locAP[j]
                dist = np.sqrt((xi - xj)**2. + (yi - yj)**2.)
                h_random = np.sqrt(np.random.normal(loc=0, scale=np.sqrt(1/2))**2 + np.random.normal(loc=0, scale=np.sqrt(1/2))**2)
                # self.h_U[i][j] = theta_U * dist**(-3) * h_random
                self.h_U[i][j] = theta_U * dist**(-3) * h_random * 125
                self.h_D[i][j] = 2 * self.h_U[i][j]

