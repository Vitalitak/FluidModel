import math as m
import matplotlib.pyplot as plt
import numpy as np

"""
four order Runge-Kutta method for solution equation
dy/dx=f(x, y)

"""
def RKPois(h, y0, Nx):

    y = [0 for k in range(0, Nx)]

    y[0] = y0



    return y

def main():

    # initialisation of parameters
    boxsize = 1000  # mkm
    dt = 0.01  # ns
    Nx = 1000
    tEnd = 50  # ns
    dne = 0.01
    dni = 0.001
    me = 1
    mi = 70000
    C = 1.4E-16
    C /= 1.6E-19
    Te = 2.3

    # Te *= 1.7E12 / 9.1  # kT/me

    Nt = int(tEnd / dt)
    dx = boxsize / Nx
    V = [0 for k in range(0, Nx)]
    ne = [1 for k in range(0, Nx)]
    ni = [1 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]
    ui = [-0.01 for k in range(0, Nx)]
    Vrf = 0
    Vdc = -10
    # ne = dist_Bolt(V, 1, Te)

    V = RKPois(dx, Vdc, Nx)

    plt.plot(V)
    plt.ylabel('V')
    plt.show()

    return 0

if __name__ == "__main__":
        main()