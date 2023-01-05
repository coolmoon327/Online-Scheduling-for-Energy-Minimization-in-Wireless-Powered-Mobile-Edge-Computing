from Parameter import Parameter
from Environment import Environment
from Algorithm import Algorithm
from scipy.io import savemat
import numpy as np

generate_Map = False    # 初次运行时，需要重新生成基站地图

# 算法编号：0-WPMEC 1-LCO 2-FO
def simulation(algID=0, V=0, N=0, M=0):
    print(f'-{algID}')
    if V:
        param = Parameter(V=V)
    elif N:
        param = Parameter(N=N)
    else:
        param = Parameter(M=M)
    if algID == 1:
        param.V = param.V / 1.5
    env = Environment(param=param)
    algorithm = Algorithm(param=param)

    if generate_Map:
        param.generateMap()
        param.saveMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')
        param.printMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.txt')
        param.loadMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')

    l1 = []  # 每个slot消耗的能量
    l2 = []  # 每个slot的队列Q的长度

    for iterator in range(param.MAX_EPISODES):
        if algID == 0:
            algorithm.executeWPMEC()
        elif algID == 1:
            algorithm.executeLCO()
        else:
            algorithm.executeFO()
        env.step()
        #print(param.E_H)

        E_T = np.sum(param.a_T * param.P_T * param.tau_T)
        E_C = 0.
        for j in range(param.M):
            for i in range(param.N):
                E_C += param.eta * param.a[i][j] * param.phi[i] * param.D_O[i]
        l1.append(E_T + E_C)
        L = np.sum(param.Q)
        l2.append(L)
        #print(param.D_O)
        #print(param.a)
    #algorithm.Show()
    print(sum(l1)/len(l1))
    print(sum(l2)/len(l2))
    # print(param.h_U)

    return sum(l1)/len(l1), sum(l2)/len(l2)


####################对比实验#######################

type = {3}  # 传入 1、2、3 对应横坐标为 V、N、M
rounds = 1

if 1 in type:
    print('step 1: change V\n')
    data_V = {'eProposed': [], 'eLCO': [], 'eFO': [], 'lProposed': [], 'lLCO': [], 'lFO': []}
    for v in range(1, 6):
        print(f'{v} ')
        for nRound in range(1, rounds+1):
            energy, latency = simulation(algID=0, V=v)     # 主算法
            data_V['eProposed'].append(energy*10)
            data_V['lProposed'].append(latency/(3.5*30))
            energy, latency = simulation(algID=1, V=v)     # 对比算法1-LCO
            data_V['eLCO'].append(energy*10)
            data_V['lLCO'].append(latency/(3.5*30))
            energy, latency = simulation(algID=2, V=v)     # 对比算法2-FO
            data_V['eFO'].append(energy*10)
            data_V['lFO'].append(latency/(3.5*30))
    savemat('change_v.mat', data_V, False)

if 2 in type:
    print('step 2: change N\n')
    data_n = {'eProposed': [], 'eLCO': [], 'eFO': [], 'lProposed': [], 'lLCO': [], 'lFO': []}
    for n in range(10, 60, 10):
        print(f'{n} ')
        for nRound in range(1, rounds+1):
            energy, latency = simulation(algID=0, N=n)    # 主算法
            data_n['eProposed'].append(energy*10)
            data_n['lProposed'].append(latency/(3.5*30))
            energy, latency = simulation(algID=1, N=n)    # 对比算法1-LCO
            data_n['eLCO'].append(energy*10)
            data_n['lLCO'].append(latency/(3.5*30))
            energy, latency = simulation(algID=2, N=n)    # 对比算法2-FO
            data_n['eFO'].append(energy*10)
            data_n['lFO'].append(latency/(3.5*30))
    savemat('change_n.mat', data_n, False)

if 3 in type:
    print('step 3: change M\n')
    data_m = {'eProposed': [], 'eLCO': [], 'eFO': [], 'lProposed': [], 'lLCO': [], 'lFO': []}
    for m in range(5, 10):
        print(f'{m} ')
        for nRound in range(1, rounds+1):
            energy, latency = simulation(algID=0, M=m)    # 主算法
            data_m['eProposed'].append(energy*10)
            data_m['lProposed'].append(latency/(3.5*30))
            energy, latency = simulation(algID=1, M=m)    # 对比算法1-LCO
            data_m['eLCO'].append(energy*10)
            data_m['lLCO'].append(latency/(3.5*30))
            energy, latency = simulation(algID=2, M=m)    # 对比算法2-FO
            data_m['eFO'].append(energy*10)
            data_m['lFO'].append(latency/(3.5*30))
    savemat('change_m.mat', data_m, False)
