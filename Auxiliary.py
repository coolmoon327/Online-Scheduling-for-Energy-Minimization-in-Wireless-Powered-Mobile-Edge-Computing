def cube_root(x):
    if x >= 0:
        return x ** (1./3.)
    else:
        return - ((-x) ** (1./3.))
