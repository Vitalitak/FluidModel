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

    n = 1 = 10^10 cm-3 = 10^-2 mkm-3
    V = n * (10^-2 mkm-3) * (dx mkm)^2 * (1.6E-19 C) / (8.85E-18 F/mkm) = n * dx^2 * 1.8E-4 V
    """
    Nx = len(ne)
    dx = boxsize / Nx
    V = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nx)]
    b = [0 for k in range(0, Nx)]

    # forward
    # boundary conditions on electrode surface: (V)e = Ve
    #a[0] = 0.5
    #b[0] = 0.5 * (ne[0] - ni[0]) * dx * dx
    a[0] = 0
    b[0] = Ve

    for i in range(1, Nx-1):
        a[i] = 1/ (2-a[i-1])
        b[i] = (b[i-1] - (ne[i] - ni[i]) * dx * dx * 0.00018)/(2-a[i-1])

    # boundary condition on plasma surface: (dV/dx)p = 0
    a[Nx-1] = 0
    #b[Nx-1] = (b[Nx-2] - (ne[Nx-1] - ni[Nx-1]) * dx * dx)/(2-a[Nx-2])
    b[Nx-1] = 0  #  (V)p = 0
    #b[Nx - 1] = b[Nx - 2] / (1 - a[Nx - 2])  # (dV/dx)p = 0

    # backward
    V[Nx-1] = b[Nx-1]
    for i in range(Nx-1, 0, -1):
        V[i-1] = a[i-1]*V[i]+b[i-1]

    return V

def momentum(V, uprev, m, boxsize, dt):

    """
    sweep method solution of momentum balance equation
    """

    Nx = len(V)
    dx = boxsize / Nx
    u = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nx)]
    b = [0 for k in range(0, Nx)]

    # forward
    # boundary conditions on electrode surface: (du/dx)e = 0
    #a[0] = -uprev[1] * dt / 4.0 / dx
    #b[0] = (V[1] - V[0])/dx/m - uprev[0]/dt
    a[0] = 1
    b[0] = 0

    for i in range(1, Nx - 1):
        a[i] = uprev[i+1] / 4.0 / dx / (-1 / dt + uprev[i - 1] * a[i-1] / 4.0 / dx)
        b[i] = (-uprev[i-1] / 4.0 / dx * b[i - 1] + (V[i+1]-V[i]) * 1.76E-1 /dx/m - uprev[i] / dt) / (-1 / dt + uprev[i-1] * a[i-1] / 4.0 / dx)

    # boundary condition on plasma surface: (du/dx)p = 0
    a[Nx - 1] = 0
    #b[Nx - 1] = (-uprev[Nx-2] / 4.0 / dx * b[Nx-2] + (V[Nx-1]-V[Nx-2])/dx - uprev[Nx-1] / dt) / (-1 / dt + uprev[Nx-2] * a[Nx-2] / 4.0 / dx)  # boundary conditions for u (u[Nx-1]-u[Nx-2])
    #b[Nx - 1] = 0  # (u)p = 0
    b[Nx - 1] = b[Nx - 2]/(1 - a[Nx - 2])  # (du/dx)p = 0

    # backward
    u[Nx - 1] = b[Nx - 1]
    for i in range(Nx - 1, 0, -1):
        u[i - 1] = a[i - 1] * u[i] + b[i - 1]

    return u

def continuity(u, nprev, dn, boxsize, dt):

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
    # boundary conditions on electrode surface: (dn/dt)e = 0
    #a[0] = u[0] / (-1/dt-(u[1]-u[0])/dx)
    #b[0] = -nprev[0] / (-1-(u[1]-u[0])*dt/dx)
    a[0] = 0
    b[0] = nprev[0]
    #b[0] = nprev[0] - dn

    for i in range(1, Nx - 1):
        a[i] = u[i] / ((-1/dt-(u[i+1]-u[i])/dx) + u[i]/2.0/dx*a[i-1])
        b[i] = (-u[i]/2.0/dx*b[i-1]-nprev[i]/dt) / ((-1/dt-(u[i+1]-u[i])/dx) + u[i]/2.0/dx*a[i-1])

    # boundary condition on plasma surface: (dn/dt)p = 0
    a[Nx - 1] = 0
    #b[Nx - 1] = (-u[Nx - 1]/2.0/dx*b[Nx-2]-nprev[Nx-1]/dt) / ((-1/dt-(u[Nx-1]-u[Nx-2])/dx) + u[Nx-1]/2.0/dx*a[Nx-2]) # boundary conditions for u (u[Nx-1]-u[Nx-2])
    #b[Nx - 1] = nprev[Nx - 1] + dn  # (dn/dt)p = dn0/dt
    b[Nx - 1] = nprev[Nx - 1]  # (n)p = np (dn/dt)p = 0

    # backward
    n[Nx - 1] = b[Nx - 1]
    for i in range(Nx - 1, 0, -1):
        n[i - 1] = a[i - 1] * n[i] + b[i - 1]

    return n

def dist_Bolt(V, np, Te):

    """
    Boltzmann distribution for electrons
    """

    Nx = len(V)
    n = [0 for k in range(0, Nx)]

    for i in range(0, Nx):
        n[i] = np * m.exp(V[i] / Te)

    """
    plt.plot(V)
    plt.ylabel('V')
    plt.show()

    plt.plot(n)
    plt.ylabel('n')
    plt.show()
    """

    return n

def main():

    """
    First block:
    self-consistent solution of Poisson equation, electrons and ions momentum balance, and
    electron and ion continuity equation

    Second block:
    Monte-Carlo simulation of ion transport across the sheath
    """

    # initialisation of parameters
    boxsize = 1000 # mkm
    dt = 0.01 # ns
    Nx = 1000
    tEnd = 0.1 # ns
    dne = 0.01
    dni = 0.001
    me = 1
    mi = 70000
    C = 1.4E-16
    C /= 1.6E-19
    Te = 2.3

    #Te *= 1.7E12 / 9.1  # kT/me

    Nt = int(tEnd/dt)
    V = [0 for k in range(0, Nx)]
    ne = [1 for k in range(0, Nx)]
    ni = [1 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]
    ui = [-0.01 for k in range(0, Nx)]
    Vrf = 0
    Vdc = -10
    ne = dist_Bolt(V, 1, Te)
    for k in range(0, 200):
        ni[k] = k/200


    for i in range(0, Nt):
        t = i*dt
        Ve = Vdc + Vrf * np.sin(0.01356*t)
        Velectron = [i * -1 for i in V]
        ne = dist_Bolt(V, 1, Te)
        V = Pois(ne, ni, Ve, boxsize)
        #Velectron = [i*-1 for i in V]
        #ue = momentum(Velectron, ue, me, boxsize, dt)
        #ue[0] = -2
        ui = momentum(V, ui, mi, boxsize, dt)
        #ne = continuity(ue, ne, dne, boxsize, dt)
        #ne = dist_Bolt(Velectron, 1, Te)
        ni = continuity(ui, ni, dni, boxsize, dt)
        Vdc += (ni[0]*ui[0] - ne[0]*ue[0]) * dt / C

    plt.plot(V)
    plt.ylabel('V')
    plt.show()

    plt.plot(ui,'r', ue, 'b')
    plt.axis([-50, Nx+50,-0.5, 0.5])
    plt.ylabel('velocity')
    plt.text(500, 1.5, r'red - ions, blue - electrons')
    plt.show()

    plt.plot(ni, 'r', ne, 'b')
    plt.axis([-50, Nx+50,-10, 10])
    plt.ylabel('concentration')
    plt.text(500, 0.5, r'red - ions, blue - electrons')
    plt.show()
    #print(ni)

    return 0

if __name__ == "__main__":
    main()