import math as m
import matplotlib.pyplot as plt
import numpy as np

"""
1D Fluid model of collisionless Ar plasma sheath for ions
Maxwell-Boltzmann distribution for electrons
Electrode potential in equivalent circuit model
"""


def RKPois(h, y0, Nx):

    """
    four order Runge-Kutta method for solution equation
    dy/dx=f(x, y)

    Poisson equation with Maxwell-Boltzmann electrons and ion concentration from fluid model
    for dn/dt<<d(nu)/dx and du/dt<<u(du/dx)

    dV/dx = (2*(1-2V)^1/2+exp(V)-15/4)^1/2


    """

    y = [0 for k in range(0, Nx)]

    y[0] = y0



    return y

def main():

    """
    First block:
    self-consistent solution of Poisson equation, ions momentum balance, and ions continuity equation
    Maxwell-Boltzmann distribution for electrons

    Second block:
    Monte-Carlo simulation of ion transport across the sheath
    """

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