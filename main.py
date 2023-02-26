import math as m
import matplotlib.pyplot as plt
import numpy as np

"""
1D Fluid model of collisionless Ar plasma sheath for electrons and ions
Electrode potential in equivalent circuit model
"""

def Pois(ne, ni, Ve, boxsize):

    """
    sweep method solution of Poisson equation
    electrode boundary condition Ve
    """
    Nx = len(ne)
    dx = boxsize / Nx
    V = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nx)]
    b = [0 for k in range(0, Nx)]

    # forward
    a[0] = 0.5
    b[0] = 0.5 * (ne[0] - ni[0]) * dx * dx

    for i in range(1, Nx-1):
        a[i] = 1/ (2-a[i-1])
        b[i] = (b[i-1] - (ne[i] - ni[i]) * dx * dx)/(2-a[i-1])

    a[Nx-1] = 0
    b[Nx-1] = (b[Nx-2] - (ne[Nx-1] - ni[Nx-1]) * dx * dx)/(2-a[Nx-2])

    # backward
    V[Nx-1] = b[Nx-1]
    for i in range(Nx-1, 0, -1):
        V[i-1] = a[i-1]*V[i]+b[i-1]

    return V

def momentum(V, uprev, boxsize, dt):

    """
    sweep method solution of momentum conservation equation
    """

    Nx = len(V)
    dx = boxsize / Nx
    u = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nx)]
    b = [0 for k in range(0, Nx)]

    # forward
    a[0] = -uprev[1] * dt / 4.0 / dx
    b[0] = (V[1] - V[0])/dx - uprev[0]/dt

    for i in range(1, Nx - 1):
        a[i] = uprev[i+1] / 4.0 / dx / (-1 / dt + uprev[i - 1] * a[i-1] / 4.0 / dx)
        b[i] = (-uprev[i-1] / 4.0 / dx * b[i - 1] + (V[i]-V[i-1])/dx - uprev[i] / dt) / (-1 / dt + uprev[i-1] * a[i-1] / 4.0 / dx)

    a[Nx - 1] = 0
    b[Nx - 1] = (-uprev[Nx-2] / 4.0 / dx * b[Nx-2] + (V[Nx-1]-V[Nx-2])/dx - uprev[Nx-1] / dt) / (-1 / dt + uprev[Nx-2] * a[Nx-2] / 4.0 / dx)  # boundary conditions for u (u[Nx-1]-u[Nx-2])

    # backward
    u[Nx - 1] = b[Nx - 1]
    for i in range(Nx - 1, 0, -1):
        u[i - 1] = a[i - 1] * u[i] + b[i - 1]

    return u

def continuity(u, nprev, boxsize, dt):

    """
    sweep method solution of continuity equation
    """

    Nx = len(nprev)
    dx = boxsize / Nx
    n = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nx)]
    b = [0 for k in range(0, Nx)]

    # forward
    a[0] = u[0] / (-1/dt-(u[1]-u[0])/dx)
    b[0] = -nprev[0] / (-1-(u[1]-u[0])*dt/dx)

    for i in range(1, Nx - 1):
        a[i] = u[i] / ((-1/dt-(u[i+1]-u[i])/dx) + u[i]/2.0/dx*a[i-1])
        b[i] = (-u[i]/2.0/dx*b[i-1]-nprev[i]/dt) / ((-1/dt-(u[i+1]-u[i])/dx) + u[i]/2.0/dx*a[i-1])

    a[Nx - 1] = 0
    b[Nx - 1] = (-u[Nx - 1]/2.0/dx*b[Nx-2]-nprev[Nx-1]/dt) / ((-1/dt-(u[Nx-1]-u[Nx-2])/dx) + u[Nx-1]/2.0/dx*a[Nx-2]) # boundary conditions for u (u[Nx-1]-u[Nx-2])

    # backward
    n[Nx - 1] = b[Nx - 1]
    for i in range(Nx - 1, 0, -1):
        n[i - 1] = a[i - 1] * n[i] + b[i - 1]

    return n

def main():

    """
    First block:
    self-consistent solution of Poisson equation, electrons and ions momentum conservation, and
    electron and ion continuity equation

    Second block:
    Monte-Carlo simulation of ion transport across the sheath
    """

    # initialisation of parameters
    boxsize = 500
    dt = 1
    Nx = 10000

    V = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ni = [1 for k in range(0, Nx)]
    u = [0 for k in range(0, Nx)]

    V = Pois(ne, ni, 0, boxsize)
    u_2 = momentum(V, u, boxsize, dt)
    u = u_2
    ni_2 = continuity(u, ni, boxsize, dt)
    ni = ni_2

    plt.plot(V)
    plt.show()
    print(u)
    #print(ni)


    return 0

if __name__ == "__main__":
    main()