import numpy as np
import Parameter

'''
reset: 重制环境
step: 继续模拟环境，进入下一个状态
'''

class Environment(object):
    def __init__(self, param: Parameter):
        super().__init__()
        self.param = param
        self.reset()

    def reset(self):
        param = self.param
        param.B_remain = [0.5 for i in range(param.N)]
        param.B_minus = param.B_max - param.B_remain
        # param.A = np.array([np.random.randint(1 * 10**4, 5 * 10**4) for _ in range(param.N)])
        param.A = np.array([np.random.rand()*0.05 for _ in range(param.N)])
        param.D_L = np.array([0. for _ in range(param.N)])
        param.D_O = np.array([0. for _ in range(param.N)])
        param.Q = np.array([0. for _ in range(param.N)])

    def step(self):
        param = self.param
        param.E_H = np.array([0. for _ in range(param.N)])
        param.A = np.array([np.random.rand()*0.5+0.1 for _ in range(param.N)])
        param.D_O = np.array([0. for _ in range(param.N)])
        for i in range(param.N):
            """能量"""
            param.E_H[i] = np.sum(param.mu[i] * param.a_T * param.P_T * param.h_D[i] * param.tau_T)
            #print("E_H", end='')
            #print(param.mu[i] * param.a_T * param.P_T * param.h_D[i] * param.tau_T)
            #print(param.E_H[i])
            param.E_L[i] = param.k[i] * param.f[i]**3 * param.tau_L[i]
            param.E_O[i] = param.P[i] * param.tau[i]
            param.B_remain[i] = max(0., min(param.B_max[i], param.B_remain[i] - param.E_L[i] - param.E_O[i] + param.E_H[i]))
            param.B_minus[i] = param.B_max[i] - param.B_remain[i]
            """队列"""
            param.D_L[i] = (param.f[i] * param.tau_L[i] / param.phi[i])
            param.D_O[i] = (np.sum(param.a[i] * param.B * param.tau[i] / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i] / param.sigma**2)))
            param.Q[i] = max(0, param.Q[i] - param.D_L[i] - param.D_O[i] + param.A[i])
            # param.h_D = np.array([[(np.random.rand()+1.5)*0.01 for _ in range(param.M)] for _ in range(param.N)])
            param.h_U = np.array([[(np.random.rand()+0.5)*0.1 for _ in range(param.M)] for _ in range(param.N)])
            param.h_D = 2.0 * param.h_U

