import math as m
import matplotlib.pyplot as plt
import numpy as np

"""
four order Runge-Kutta method for solution equation
dy/dx=f(x, y)

Poisson equation with Maxwell-Boltzmann electrons and ion concentration from fluid model
for dn/dt = 0 and du/dt = 0

dKsi/dx=F(x, Ksi)

"""
def RKPois(dx, Nx, Ksi, Npl, n0, Ti, Te, V0):

    e = 1.6E-19
    eps0 = 8.85E-12
    kTe = Te * 1.6E-19  # J

    """
    dKsi/dx=F(x, Ksi)
    F = -(C+A*exp(Ksi)+B*(4/3-2Te/(3Ti)*Ksi)^3/2)^1/2
    
    C=((Ksi[Npl-1]-Ksi[Npl-2])/dx)^2-2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)*m.exp(Ksi[Npl-1])-2*e*e*n0/eps0/kTe*Ti/Te*(4/3-2*Te/3/Ti*Ksi[Npl-1])^1.5
    A=2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)
    B=2*e*e*n0/eps0/kTe*Ti/Te
    Ksi(a)=Ksi[Npl-1]
    
    Four order Runge-Kutta method
    f1=F(x[n], Ksi[n])
    f2=F(x[n]+dx/2, Ksi[n]+dx/2*f1) = F(x[n]+dx/2, Ksi[n]+dx/2*F(x[n], Ksi[n]))
    f3=F(x[n]+dx/2, Ksi[n]+dx/2*f2)
    f4=F(x[n]+dx, Ksi[n]+dx*f3)
    
    Ksi[n+1]=Ksi[n]+dx/6*(f1+2*f2+2*f3+f4)
    
    """

    #dx = x[Npl - 1]-x[Npl - 2]
    #Nx = len[Ksi]

    C = m.pow((Ksi[Npl - 1] - Ksi[Npl - 2]) / dx, 2) - 2 * e*e * n0 / eps0 / kTe * m.exp(e * V0 / kTe) * m.exp(
        Ksi[Npl - 1]) - 2 * e*e * n0 / eps0 / kTe * Ti / Te * m.pow((4 / 3 - 2 * Te / 3 / Ti * Ksi[Npl - 1]), 1.5)
    A = 2 * e*e * n0 / eps0 / kTe * m.exp(e * V0 / kTe)
    B = 2 * e*e * n0 / eps0 / kTe * Ti / Te

    #print(C)
    #print(A)

    for i in range(Npl-1, Nx-1):
        f1=-m.pow((C+A*m.exp(Ksi[i])+B*m.pow((4/3-2*Te/(3*Ti)*Ksi[i]), 1.5)), 0.5)
        #print(f1)
        f2=-m.pow((C+A*m.exp(Ksi[i]+dx/2*f1)+B*m.pow((4/3-2*Te/(3*Ti)*(Ksi[i]+dx/2*f1)), 1.5)), 0.5)
        #print(f2)
        f3 = -m.pow((C + A * m.exp(Ksi[i] + dx / 2 * f2) + B * m.pow(
            (4 / 3 - 2 * Te / (3 * Ti) * (Ksi[i] + dx / 2 * f2)), 1.5)), 0.5)
        print(f3)
        f4=-m.pow((C+A*m.exp(Ksi[i]+dx*f3)+B*m.pow((4/3-2*Te/(3*Ti)*(Ksi[i]+dx*f3)), 1.5)), 0.5)
        print(f4)
        Ksi[i+1] = Ksi[i]+dx/6*(f1+2*f2+2*f3+f4)



    return Ksi

def main():

    # initialisation of parameters
    boxsize = 4E-4  # m
    dt = 0.1  # ns
    Nx = 100000
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
    e = 1.6E-19
    n0 = 1E16  # m-3
    a = 1.3E-4  # m
    P = 1.4  #  P = ni(a)/n0
    eps0 = 8.85E-12
    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J
    #V0 = -0.01
    V0 = kTe / e * (1 - P) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * a) - 1)

    # Te *= 1.7E12 / 9.1  # kT/me

    Nt = int(tEnd / dt)
    dx = boxsize / Nx

    x = [k*dx for k in range(0, Nx)]

    Vpl = [0 for k in range(0, Nx)]
    Nipl = [0 for k in range(0, Nx)]
    Nepl = [0 for k in range(0, Nx)]
    uipl = [0 for k in range(0, Nx)]
    Epl = [0 for k in range(0, Nx)]
    dKsidxpl = [0 for k in range(0, Nx)]

    #V0 = [0 for k in range(0, Nx)]

    Npl = int(a/dx)

    for i in range(0, Npl):
        #V0[i] = kTe / e * (1 - 1.4) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * (x[i]+dx)) - 1)
        Vpl[i] = V0*(1+2*Ti/Te*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1))
        Nipl[i] = n0 - n0*e*V0/kTe*(m.cosh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])-1)
        uipl[i] = n0 * m.sqrt(kTi/mi) / Nipl[i]
        Nepl[i] = n0 * m.exp(e*Vpl[i]/kTe)
        dKsidxpl[i] = 2*Ti/Te*e*V0/kTe*m.sqrt(e*e*n0/2/eps0/kTi)*m.sinh(m.sqrt(e*e*n0/2/eps0/kTi)*x[i])
        Epl[i] = -kTe/e*dKsidxpl[i]


    print(m.sqrt(eps0*kTe/e*e*n0))
    print(V0)


    Ksi = [0 for k in range(0, Nx)]
    #Ne = [0 for k in range(0, Nx)]
    Ni = [0 for k in range(0, Nx)]
    #Ve = [0 for k in range(0, Nx)]
    #Vi = [0 for k in range(0, Nx)]
    Vrf = 0

    for i in range(0, Npl):
        Ksi[i] = e*(Vpl[i]-V0)/kTe
        Ni[i] = Nipl[i]/n0

    plt.plot(x, dKsidxpl)
    plt.ylabel('dKsi/dx')
    plt.show()

    Ksi = RKPois(dx, Nx, Ksi, Npl, n0, Ti, Te, V0)

    """
    plt.plot(x, Ksi)
    plt.ylabel('Ksi')
    plt.show()

    plt.plot(x, Ni)
    plt.ylabel('N')
    plt.show()
    """

    plt.plot(x, Nepl, 'b')
    plt.plot(x, Nipl, 'r')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, Ksi)
    plt.ylabel('Ksi')
    plt.show()

    plt.plot(x, Epl)
    plt.ylabel('E')
    plt.show()

    plt.plot(x, uipl)
    plt.ylabel('u')
    plt.show()


    return 0

if __name__ == "__main__":
        main()