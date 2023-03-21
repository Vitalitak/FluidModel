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
    boxsize = 1.36E-4  # m
    dt = 0.1  # ns
    Nx = 1000000
    tEnd = 50  # ns
    dne = 0.01
    dni = 0.001
    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    C = 1.4E-16
    C /= 1.6E-19
    Te = 2.3  # eV
    Ti = 0.06  # eV
    Vdc = -18
    #V0 = -0.01
    e = 1.6E-19
    n0 = 1E16  # m-3
    eps0 = 8.85E-12
    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J
    V0 = -0.01
    #V0 = -1/2*m.sqrt(Te/2/Ti)*kTe/e/m.sinh(m.sqrt(e*e*n0/2/eps0/kTi)*boxsize-m.sqrt(Te/2/Ti))

    # Te *= 1.7E12 / 9.1  # kT/me

    Nt = int(tEnd / dt)
    dx = boxsize / Nx

    x = [k*dx for k in range(0, Nx)]

    Vpl = [0 for k in range(0, Nx)]
    Nipl = [0 for k in range(0, Nx)]
    Nepl = [0 for k in range(0, Nx)]
    uipl = [0 for k in range(0, Nx)]
    dKsidx = [0 for k in range(0, Nx)]
    for i in range(0, Nx):
        Vpl[i] = V0*(1+2*Ti/Te*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1))
        Nipl[i] = n0 - n0*e*V0/kTe*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1)
        uipl[i] = n0 * m.sqrt(kTi/mi) / Nipl[i]
        Nepl[i] = n0 * m.exp(e*Vpl[i]/kTe)
        dKsidx[i] = 2*Ti/Te*e*V0/kTe*m.sqrt(e*e*n0/2/eps0/kTi)*m.sinh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])


    print(m.sqrt(eps0*kTe/e*e*n0))
    print(V0)

    V = [0 for k in range(0, Nx)]
    ne = [1 for k in range(0, Nx)]
    ni = [1 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]
    ui = [-0.01 for k in range(0, Nx)]
    Vrf = 0

    # ne = dist_Bolt(V, 1, Te)

    V = RKPois(dx, Vdc, Nx)

    plt.plot(x, Nepl, 'b')
    plt.plot(x, Nipl, 'r')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, Vpl)
    plt.ylabel('V')
    plt.show()

    plt.plot(x, dKsidx)
    plt.ylabel('E')
    plt.show()

    return 0

if __name__ == "__main__":
        main()