import math as m
import matplotlib.pyplot as plt
import numpy as np

"""
four order Runge-Kutta method for solution equation
dy/dx=f(x, y)

Poisson equation with Maxwell-Boltzmann electrons and ion concentration from fluid model
for dn/dt<<d(nu)/dx and du/dt<<u(du/dx)

dV/dx = (2*(1-2V)^1/2+exp(V)-15/4)^1/2

"""
def RKPois(h, y0, Nx):

    y = [0 for k in range(0, Nx)]

    y[0] = y0



    return y

def main():

    # initialisation of parameters
    boxsize = 5E-7  # m
    dt = 0.01  # ns
    Nx = 10000
    tEnd = 50  # ns
    dne = 0.01
    dni = 0.001
    me = 1
    mi = 73000
    C = 1.4E-16
    C /= 1.6E-19
    Te = 2.3  # eV
    Ti = 0.06  # eV
    Vdc = -18
    V0 = -0.01
    e = 1.6E-19
    n0 = 1E16  # m-3
    eps0 = 8.85E-12
    kTi = Ti * 1.38E-23  # J
    kTe = Te * 1.38E-23  # J

    # Te *= 1.7E12 / 9.1  # kT/me

    Nt = int(tEnd / dt)
    dx = boxsize / Nx

    x = [k*dx for k in range(0, Nx)]

    Vpl = [0 for k in range(0, Nx)]
    Npl = [0 for k in range(0, Nx)]
    for i in range(0, Nx):
        Vpl[i] = V0*(1+2*Ti/Te*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1))
        Npl[i] = n0 - e*V0/kTe*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1)

    V = [0 for k in range(0, Nx)]
    ne = [1 for k in range(0, Nx)]
    ni = [1 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]
    ui = [-0.01 for k in range(0, Nx)]
    Vrf = 0

    # ne = dist_Bolt(V, 1, Te)

    V = RKPois(dx, Vdc, Nx)

    plt.plot(x, Vpl)
    plt.ylabel('V')
    plt.show()

    return 0

if __name__ == "__main__":
        main()