import numpy as np
import scipy as sp
from scipy.optimize import linear_sum_assignment, root, fsolve, least_squares
import Parameter
from Auxiliary import cube_root

'''
execute: 执行算法
'''

ABS_TOL = 0.3 # absolute tolerance in terms of float precision


class Algorithm(object):
    def __init__(self, param: Parameter):
        super().__init__()
        self.param = param

    def WPT(self):
        param = self.param
        param.c_T = np.array([0. for _ in range(param.M)])
        param.a_T = np.array([0 for _ in range(param.M)])
        param.P_T = np.array([0. for _ in range(param.M)])
        param.tau_T = np.array([0. for _ in range(param.M)])

        for j in range(param.M):
            sum = 0.
            for i in range(param.N):
                sum += param.B_minus[i] * param.B_scalar * param.mu[i] * param.h_D[i][j]
            param.c_T[j] = param.V - sum

        def getArgmin(matrix):
            return np.argmin(matrix), np.min(matrix)

        (j_star, cp_min) = getArgmin(param.c_T * param.P_T_max)
        if cp_min < 0:
            param.a_T[j_star] = 1
            param.P_T[j_star] = param.P_T_max[j_star]
            param.tau_T[j_star] = param.T

    def LocalComputation(self):
        param = self.param
        param.f = np.array([0. for _ in range(param.N)])
        for i in range(param.N):
            if param.B_minus[i] == 0:
                param.f[i] = param.f_max[i]
            else:
                param.f[i] = np.min([param.f_max[i], np.sqrt(param.Q[i] * param.Q_scalar / (3 * param.k[i] * param.phi[i] * param.B_minus[i] * param.B_scalar))])
        # print(param.f)

    def ComputationOffloading(self):
        param = self.param
        # 计算a的初值与对应的tau
        param.a = np.array([[0 for _ in range(param.M)] for _ in range(param.N)])
        for i in range(param.N):
            param.a[i][np.random.randint(0, param.M)] = 1
        connected_wd = np.sum(param.a, axis=0)
        for i in range(param.N):
            j_temp = np.argwhere(param.a[i] == 1)
            param.tau[i] = param.T / connected_wd[j_temp]

        round = 0
        while round < 3:
            # 计算P
            for i in range(param.N):
                j_temp = np.argwhere(param.a[i] == 1)
                if param.phi[i] * param.eta > param.Q[i] * param.Q_scalar or param.B_minus[i] == 0:
                    param.P[i] = param.P_max[i]
                else:
                    param.P[i] = max(0., min(param.P_max[i], (((param.Q[i] * param.Q_scalar - param.phi[i] * param.eta) * param.B) / (param.B_minus[i] * param.B_scalar * param.v[i] * np.log(2))) - ( param.sigma[j_temp]**2 / param.h_U[i][j_temp])))

            # 计算时间分配
            param.tau = np.array([param.T for _ in range(param.N)])
            # 匈牙利算法求解
            cost = np.array([[0. for _ in range(param.M)] for _ in range(param.N)])
            for i in range(param.N):
                for j in range(param.M):
                    cost[i][j] = (param.phi[i] * param.eta - param.Q[i] * param.Q_scalar) * param.B * param.T / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i][j] / param.sigma[j]**2)
            row_id, col_id = linear_sum_assignment(cost)

            # 设置本次迭代中计算出的a矩阵与对应的tau
            param.a = np.array([[0 for _ in range(param.M)] for _ in range(param.N)])
            for i, j in zip(row_id, col_id):
                param.a[i][j] = 1
            connected_wd = np.sum(param.a, axis=0)
            for i in range(param.N):
                j_temp = np.argwhere(param.a[i] == 1)
                if len(j_temp) < 1:
                    param.tau[i] = 0.
                else:
                    if connected_wd[j_temp] == 0:
                        param.tau[i] = param.T
                    else:
                        param.tau[i] = param.T / connected_wd[j_temp]

            round += 1

    def AdjustVariableValues(self):
        param = self.param
        j_wpt = np.argwhere(param.a_T == 1)
        N_star = set()
        N_rest = set()
        for i in range(param.N):
            if param.a[i][j_wpt] == 1:
                N_star.add(i)
            else:
                N_rest.add(i)

        def f1(p, i):
            # 式16计算的f
            # f = f1(p) according to the marginal energy efficiency equation
            temp = np.sum(param.a[i] * param.v[i] * np.log(2) / (3 * param.k[i] * param.phi[i] * param.B) * (param.sigma**2 / param.h_U[i] + p)) + param.eta / (3 * param.k[i])
            if temp <= 0:
                return 0.
            else:
                return np.sqrt(temp)

        def f2(p, tau, i):
            # 式17计算的f
            # f = f2(p, tau) according to the energy constraint (2)
            temp = -(p * tau - param.B_remain[i]) / (param.k[i] * param.T)
            return cube_root(temp)
            # if temp <= 0:
                # return 0.
            # else:
                # # print("temp:", end='')
                # # print(temp)
                # return cube_root(temp)

        def f3(p, tau, i):
            # f = f3(p, tau) according to the queueing data constraint (3)
            temp = param.phi[i] / param.T * ( param.Q[i] * param.Q_scalar - np.sum(param.a[i] * param.B * tau / param.v[i] * np.log2(1 + p * param.h_U[i] / (param.sigma**2))))
            return temp
            # if temp <= 0:
                # return 0.
            # else:
                # return temp

        def func1(p, tau, i):
            # print("f1 - f2:", end='')
            # print(f1(p, i) - f2(p, tau, i))
            return f1(p, i) - f2(p, tau, i)

        def func3(p, tau, i):
            return f1(p, i) - f3(p, tau, i)

        for i in N_rest:
            E_L_i = param.k[i] * param.f[i]**3 * param.tau_L[i]
            E_O_i = param.P[i] * param.tau[i]
            D_L_i = param.f[i] * param.tau_L[i] / param.phi[i]
            D_O_i = np.sum(param.a[i] * param.B * param.tau[i] / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i] / param.sigma**2))
            if E_L_i + E_O_i <= param.B_remain[i] and D_L_i + D_O_i <= param.Q[i]:
                continue
            param.P[i] = least_squares(fun=func1, x0=0., args=(param.tau[i], i), bounds=(0., param.P_max[i])).x
            param.f[i] = f2(param.P[i], param.tau[i], i)
            D_L_i = param.f[i] * param.tau_L[i] / param.phi[i]
            D_O_i = np.sum(param.a[i] * param.B * param.tau[i] / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i] / param.sigma**2))
            if D_L_i + D_O_i > param.Q[i]:
                # param.P[i] = max(0., fsolve(func=func3, x0=0.1, args=(param.tau[i], i)))
                param.P[i] = least_squares(fun=func3, x0=0., args=(param.tau[i], i), bounds=(0, param.P_max[i])).x
                param.f[i] = f3(param.P[i], param.tau[i], i)
                D_L_i = param.f[i] * param.tau_L[i] / param.phi[i]
                D_O_i = np.sum(param.a[i] * param.B * param.tau[i] / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i] / param.sigma**2))
                assert D_L_i + D_O_i <= param.Q[i] + ABS_TOL

        def func2(tau, i):
            # p = max(0., fsolve(func=func1, x0=0.1, args=(tau, i)))
            p = least_squares(fun=func1, x0=0., args=(tau, i), bounds=(0, param.P_max[i])).x
            # 代入式19
            ans1 = (param.V * param.eta * param.phi[i] - param.Q[i] * param.Q_scalar) * param.B / param.v[i] * np.log2(1 + p * param.h_U[i][j_wpt] / param.sigma[j_wpt]**2) + param.B_minus[i] * param.B_scalar * p
            # 让该函数的值等于c_T[j_wpt] * P_T_max[j_wpt]
            ans2 = param.c_T[j_wpt] * param.P_T_max[j_wpt]
            return np.ravel(ans1 - ans2)

        def func4(tau, i):
            # p = max(0., fsolve(func=func3, x0=0.1, args=(tau, i)))
            p = least_squares(fun=func1, x0=0., args=(tau, i), bounds=(0, param.P_max[i])).x
            # 代入式19
            ans1 = (param.V * param.eta * param.phi[i] - param.Q[i] * param.Q_scalar) * param.B / param.v[i] * np.log2(1 + p * param.h_U[i][j_wpt] / param.sigma[j_wpt]**2) + param.B_minus[i] * param.B_scalar * p
            # 让该函数的值等于c_T[j_wpt] * P_T_max[j_wpt]
            ans2 = param.c_T[j_wpt] * param.P_T_max[j_wpt]
            return np.ravel(ans1 - ans2)

        for i in N_star:
            # tau_new = fsolve(func=func2, x0=0.1, args=i)
            tau_new = least_squares(fun=func2, x0=0., args=(i,), bounds=(0, param.T)).x
            if tau_new < param.tau[i]:
                param.tau[i] = tau_new
            # param.P[i] = max(0., fsolve(func=func1, x0=0.1, args=(param.tau[i], i)))
            param.P[i] = least_squares(fun=func1, x0=0., args=(param.tau[i], i), bounds=(0, param.P_max[i])).x
            param.f[i] = f2(param.P[i], param.tau[i], i)

            D_L_i = param.f[i] * param.tau_L[i] / param.phi[i]
            D_O_i = np.sum(param.a[i] * param.B * param.tau[i] / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i] / param.sigma**2))
            if D_L_i + D_O_i > param.Q[i]:
                # tau_new = fsolve(func=func4, x0=0.1, args=i)
                tau_new = least_squares(fun=func4, x0=0., args=(i,), bounds=(0, param.T)).x
                if tau_new < param.tau[i]:
                    param.tau[i] = tau_new
                # param.P[i] = max(0., fsolve(func=func3, x0=0.1, args=(param.tau[i], i)))
                param.P[i] = least_squares(fun=func3, x0=0., args=(param.tau[i], i), bounds=(0, param.P_max[i])).x
                param.f[i] = f3(param.P[i], param.tau[i], i)
            '''
            print("marginal cost of wpt:", end='')
            print(param.c_T[j_wpt] * param.P_T_max[j_wpt])
            print("marginal cost of offloading, WD %d" % (i), end='')
            print((param.V * param.eta * param.phi[i] - param.Q[i]) * param.B / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i][j_wpt] / param.sigma[j_wpt]**2) + param.B_minus[i] * param.B_scalar * param.P[i])
            print(tau_new)
            D_L_i = param.f[i] * param.tau_L[i] / param.phi[i]
            D_O_i = np.sum(param.a[i] * param.B * param.tau[i] / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i] / param.sigma**2))
            assert D_L_i + D_O_i <= param.Q[i] + ABS_TOL
            '''
            
        
        for i in range(param.N):
            param.P[i] = max(0., min(param.P[i], param.P_max[i]))
            param.f[i] = max(0., min(param.f[i], param.f_max[i]))
            
        param.tau_T[j_wpt] = param.T - sum([param.tau[i] for i in N_star])
        #print(param.tau_T)

    def CheckConstraints(self):
        param = self.param
        assert np.sum(param.a_T) <= 1
        for i in range(param.N):
            E_L_i = param.k[i] * param.f[i]**3 * param.tau_L[i]
            E_O_i = param.P[i] * param.tau[i]
            assert E_L_i + E_O_i <= param.B_remain[i] + ABS_TOL
        for i in range(param.N):
            D_L_i = param.f[i] * param.tau_L[i] / param.phi[i]
            D_O_i = np.sum(param.a[i] * param.B * param.tau[i] / param.v[i] * np.log2(1 + param.P[i] * param.h_U[i] / param.sigma**2))
            assert D_L_i + D_O_i <= param.Q[i] + ABS_TOL
        for i in range(param.N):
            assert np.sum(param.a[i]) <= 1
        for j in range(param.M):
            assert param.a_T[j] * param.tau_T[j] + np.sum(np.multiply(np.transpose(param.a)[j], param.tau)) <= param.T + ABS_TOL
        for j in range(param.M):
            assert 0 <= param.tau_T[j] <= param.T
            assert 0 <= param.P_T[j] <= param.P_T_max[j]
        for i in range(param.N):
            assert 0 <= param.P[i] <= param.P_max[i]
            assert 0 <= param.tau[i] <= param.T
            assert 0 <= param.f[i] <= param.f_max[i]
    
    def Show(self):
        print("B_remain:", end=''); print(self.param.B_remain);
        print("q_T:", end=''); print(self.param.a_T * self.param.P_T * self.param.tau_T);
        print(self.param.a_T); print(self.param.tau_T);
        print("Q:", end=''); print(self.param.Q);
        print("P:", end=''); print(self.param.P);
        print("f:", end=''); print(self.param.f);
        
    '''主算法'''
    def executeWPMEC(self):
        self.WPT()
        self.LocalComputation()
        self.ComputationOffloading()
        self.AdjustVariableValues()
        self.CheckConstraints()
        # self.Show()

    '''对比算法1-LCO'''
    def executeLCO(self):
        self.WPT()
        self.LocalComputation()
        for i in range(self.param.N):
            self.param.f[i] = min(self.param.f[i], self.param.Q[i] * self.param.phi[i] / self.param.T)
        self.CheckConstraints()

    '''对比算法2-FO'''
    def executeFO(self):
        self.param.f_max = np.array([0 for _ in range(self.param.N)])
        self.WPT()
        self.LocalComputation()
        self.ComputationOffloading()
        self.AdjustVariableValues()
        # self.param.f = np.array([0. for _ in range(self.param.N)])
        self.CheckConstraints()

