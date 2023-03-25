import math as m
import matplotlib.pyplot as plt
import numpy as np

"""
four order Runge-Kutta method for solution equation
dy/dx=f(x, y)

Poisson equation with Maxwell-Boltzmann electrons and ion concentration from fluid model
for dn/dt = 0 and du/dt = 0

N=3
gammai = 5/3

dKsi/dx=F(x, Ksi)

"""


def RKPois1(dx, Ksi, Npl, n0, Ti, Te, V0):
    e = 1.6E-19
    eps0 = 8.85E-12
    kTe = Te * 1.6E-19  # J

    """
    Ksi(0)=0
    dKsi/dx(0) = 0
    dKsi/dx>0
    
    boundary N(x) = 0.995

    dKsi/dx=F(x, Ksi)
    F = -(A*exp(Ksi)+B*Ksi+C*(1-19*Te/2/Ti*Ksi)^3/2+D)^1/2

    A=2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)
    B=-32/19*e*e*n0/eps0/kTe
    C=8/361*Ti/Te*e*e*n0/eps0/kTe
    D=-2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)-8/361*Ti/Te*e*e*n0/eps0/kTe
    Ksi(0)=0

    Four order Runge-Kutta method
    f1=F(x[n], Ksi[n])
    f2=F(x[n]+dx/2, Ksi[n]+dx/2*f1)
    f3=F(x[n]+dx/2, Ksi[n]+dx/2*f2)
    f4=F(x[n]+dx, Ksi[n]+dx*f3)

    Ksi[n+1]=Ksi[n]+dx/6*(f1+2*f2+2*f3+f4)

    """

    # dx = x[Npl - 1]-x[Npl - 2]
    # Nx = len[Ksi]

    Ksi[2] = dx * dx * e * e * n0 / eps0 / kTe * (m.exp(e * V0 / kTe) - 1)
    A = 2 * e * e * n0 / eps0 / kTe * m.exp(e * V0 / kTe)
    B = -32/19 * e * e * n0 / eps0 / kTe
    C = 8/361 * Ti / Te * e * e * n0 / eps0 / kTe
    D = -2 * e * e * n0 / eps0 / kTe * m.exp(e * V0 / kTe) - 8/361 * Ti / Te * e * e * n0 / eps0 / kTe

    # print(A)
    # print(B)
    # print(C)
    # print(D)

    for i in range(2, Npl):
        f1 = -m.pow((A * m.exp(Ksi[i]) + B * Ksi[i] + C * m.pow((1 - 19 * Te / 2 / Ti * Ksi[i]), 1.5) + D), 0.5)
        f2 = -m.pow((A * m.exp(Ksi[i] + dx / 2 * f1) + B * (Ksi[i] + dx / 2 * f1) + C * m.pow(
            (1 - 19 * Te / 2 / Ti * (Ksi[i] + dx / 2 * f1)), 1.5) + D), 0.5)
        f3 = -m.pow((A * m.exp(Ksi[i] + dx / 2 * f2) + B * (Ksi[i] + dx / 2 * f2) + C * m.pow(
            (1 - 19 * Te / 2 / Ti * (Ksi[i] + dx / 2 * f2)), 1.5) + D), 0.5)
        f4 = -m.pow((A * m.exp(Ksi[i] + dx * f3) + B * (Ksi[i] + dx * f3) + C * m.pow(
            (1 - 19 * Te / 2 / Ti * (Ksi[i] + dx * f3)), 1.5) + D), 0.5)
        Ksi[i + 1] = Ksi[i] + dx / 6 * (f1 + 2 * f2 + 2 * f3 + f4)

    return Ksi


def RKPoisN(dx, Ksi, Nsh, Nx, n0, Ti, Te, V0):
    e = 1.6E-19
    eps0 = 8.85E-12
    kTe = Te * 1.6E-19  # J

    """
    Ksi(l-ld)=-(1-Ti/Te)
    dKsi/dx(l-ld) = 1/2/ld
    dKsi/dx<0

    dKsi/dx=-F(x, Ksi)
    F = -(A*exp(Ksi)+B*(1-Te/(4*Ti)*Ksi)^1/2+C)^1/2

    A=2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)
    B=32*m.sqrt(2/3)*Ti/Te*e*e*n0/eps0/kTe
    C=m.pow(1/2/m.sqrt(eps0*kTe/e*e*n0), 2)-2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)*m.exp(-(1-Ti/Te))-32*m.sqrt(2/3)*Ti/Te*e*e*n0/eps0/kTe*m.pow((1+Te/4/Ti*(1-Ti/Te)), 0.5)
    Ksi(a)=Ksi[Npl-1]

    Four order Runge-Kutta method
    f1=-F(x[n], Ksi[n])
    f2=-F(x[n]+dx/2, Ksi[n]+dx/2*f1)
    f3=-F(x[n]+dx/2, Ksi[n]+dx/2*f2)
    f4=-F(x[n]+dx, Ksi[n]+dx*f3)

    Ksi[n+1]=Ksi[n]+dx/6*(f1+2*f2+2*f3+f4)
    """

    # dx = x[Npl - 1]-x[Npl - 2]
    # Nx = len[Ksi]

    Ksi[Nsh] = -(1-Ti/Te)

    A = 2 * e * e * n0 / eps0 / kTe * m.exp(e * V0 / kTe)
    B = 32*m.sqrt(2/3)*Ti/Te*e*e*n0/eps0/kTe
    C = m.pow(1/2/m.sqrt(eps0*kTe/e*e*n0), 2) - 2 * e * e * n0 / eps0 / kTe *m.exp(e*V0/kTe) *m.exp(-(1-Ti/Te)) - 32*m.sqrt(2/3) * Ti / Te * e * e * n0 / eps0 / kTe *m.pow((1+Te/4/Ti*(1-Ti/Te)), 0.5)

    # print(A)
    # print(B)
    # print(C)
    # print(D)

    for i in range(Nsh, Nx - 1):
        f1 = -m.pow((A * m.exp(Ksi[i]) + B * m.pow((1 - Te / 4 / Ti * Ksi[i]), 0.5) + C), 0.5)
        f2 = -m.pow((A * m.exp(Ksi[i] + dx / 2 * f1) + B * m.pow(
            (1 - Te / 4 / Ti * (Ksi[i] + dx / 2 * f1)), 0.5) + C), 0.5)
        f3 = -m.pow((A * m.exp(Ksi[i] + dx / 2 * f2) + B * m.pow(
            (1 - Te / 4 / Ti * (Ksi[i] + dx / 2 * f2)), 0.5) + C), 0.5)
        f4 = -m.pow((A * m.exp(Ksi[i] + dx * f3) + B * m.pow(
            (1 - Te / 4 / Ti * (Ksi[i] + dx * f3)), 0.5) + C), 0.5)
        Ksi[i + 1] = Ksi[i] + dx / 6 * (f1 + 2 * f2 + 2 * f3 + f4)

    return Ksi


def main():
    # initialisation of parameters
    boxsize = 5E-5  # m
    dt = 0.1  # ns
    Nx = 500000
    tEnd = 50  # ns

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.3  # eV
    Ti = 0.06  # eV
    n0 = 1E16  # m-3
    Vdc = -18
    C = 1.4E-16
    C /= 1.6E-19

    # stitching parameters
    a = 3.5E-5  # m
    P = 0.995  # P = ni(a)/n0 boundary N(x)

    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J
    V0 = -Ti
    # V0 = kTe / e * (1 - P) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * a) - 1)

    Nt = int(tEnd / dt)
    dx = boxsize / Nx

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]

    Vpl = [0 for k in range(0, Nx)]
    Nipl = [0 for k in range(0, Nx)]
    Nepl = [0 for k in range(0, Nx)]
    uipl = [0 for k in range(0, Nx)]
    Epl = [0 for k in range(0, Nx)]
    dKsidxpl = [0 for k in range(0, Nx)]

    # V0 = [0 for k in range(0, Nx)]

    Npl = int(a / dx)

    ld = m.sqrt(eps0*kTe/e*e*n0)
    Nsh = int(Nx-ld/dx)

    """
    for i in range(0, Npl):
        #V0[i] = kTe / e * (1 - 1.4) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * (x[i]+dx)) - 1)
        Vpl[i] = V0*(1+2*Ti/Te*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1))
        Nipl[i] = n0 - n0*e*V0/kTe*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1)
        uipl[i] = n0 * m.sqrt(kTi/mi) / Nipl[i]
        Nepl[i] = n0 * m.exp(e*Vpl[i]/kTe)
        dKsidxpl[i] = 2*Ti/Te*e*V0/kTe*m.sqrt(e*e*n0/2/eps0/kTi)*m.sinh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])
        Epl[i] = -kTe/e*dKsidxpl[i]

    """
    print(Nsh)


    Ksi = [0 for k in range(0, Nx)]
    # Ne = [0 for k in range(0, Nx)]
    Ni = [0 for k in range(0, Nx)]
    # Ve = [0 for k in range(0, Nx)]
    # Vi = [0 for k in range(0, Nx)]
    Vrf = 0
    """
    for i in range(0, Npl):
        Ksi[i] = e*(Vpl[i]-V0)/kTe
        Ni[i] = Nipl[i]/n0
    """
    """
    plt.plot(x, dKsidxpl)
    plt.ylabel('dKsi/dx')
    plt.show()
    """

    Ksi = RKPois1(dx, Ksi, Npl, n0, Ti, Te, V0)

    for i in range(0, Nx):
        Ni[i] = 1 + 3 / 19 * (1 - m.sqrt(1 - 19 * Te / 2 / Ti * Ksi[i]))

    Ksi = RKPoisN(dx, Ksi, Nsh, Nx, n0, Ti, Te, V0)

    for i in range(Nsh, Nx):
        Ni[i] = 2 *m.sqrt(2/3) * m.pow((1 - Te / 4 / Ti * Ksi[i]), -0.5)

    # return to V, n
    for i in range(0, Nx):
        V[i] = kTe / e * Ksi[i] + V0
        ni[i] = n0 * Ni[i]
        ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]
        ne[i] = n0 * m.exp(e * V[i] / kTe)

    plt.plot(x, Ksi)
    plt.ylabel('Ksi')
    plt.show()

    plt.plot(x, Ni)
    plt.ylabel('N')
    plt.show()


    plt.plot(x, ne, 'b')
    plt.plot(x, ni, 'r')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, V)
    plt.ylabel('V')
    plt.show()

    # plt.plot(x, Epl)
    # plt.ylabel('E')
    # plt.show()

    plt.plot(x, ui)
    plt.ylabel('u')
    plt.show()

    return 0


if __name__ == "__main__":
    main()