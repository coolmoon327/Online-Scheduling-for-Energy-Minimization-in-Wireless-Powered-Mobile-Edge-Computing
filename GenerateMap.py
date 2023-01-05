from Parameter import Parameter

'''仅用于生成MAP文件'''

for n in range(10, 60, 10):
    param = Parameter(N=n)
    param.generateMap()
    param.saveMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')
    param.printMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.txt')

for m in range(1, 11):
    param = Parameter(M=m)
    param.generateMap()
    param.saveMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')
    param.printMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.txt')

