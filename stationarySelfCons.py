import math as m
import matplotlib.pyplot as plt
import numpy as np


"""
E = -dV/dx
Ni = ni/n0
Ne = ne/n0
Ui = ui / sqrt(gammai*kTi/mi) 
Ue = ue / sqrt(gammae*kTe/me) 

Fluid model for ions and electrons
Self-consistent system of Poisson equation momentum conservation and continuity equations
Ions have collisions

four order Runge-Kutta method for solution system
dV/dx=k(x, V, E, Ni, Ne, Ui, Ue)
dE/dx=l(x, V, E, Ni, Ne, Ui, Ue)
dNi/dx=p(x, V, E, Ni, Ne, Ui, Ue)
dNe/dx=h(x, V, E, Ni, Ne, Ui, Ue)
dUi/dx=m(x, V, E, Ni, Ne, Ui, Ue)
dUe/dx=f(x, V, E, Ni, Ne, Ui, Ue)

for dn/dt = 0 and du/dt = 0


"""


def RungeKuttasystem(Nx, dx, n0, Te, Ti, Vl, gammai, gammae, nui, nue, nuiz):
    e = 1.6E-19
    eps0 = 8.85E-12
    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    kTe = Te * 1.6E-19  # J
    kTi = Ti * 1.6E-19  # J

    """
    V(0) -> 0
    E(0) -> 0
    Ni(0) = Ne(0) = 1
    Ui(0) -> 1
    Ue(0) -> 1

    dV/dx=k(x, V, E, Ni, Ne, Ui, Ue)
    dE/dx=l(x, V, E, Ni, Ne, Ui, Ue)
    dNi/dx=p(x, V, E, Ni, Ne, Ui, Ue)
    dNe/dx=h(x, V, E, Ni, Ne, Ui, Ue)
    dUi/dx=m(x, V, E, Ni, Ne, Ui, Ue)
    dUe/dx=f(x, V, E, Ni, Ne, Ui, Ue)

    k(x, V, E, Ni, Ne, Ui, Ue) = -E
    l(x, V, E, Ni, Ne, Ui, Ue) = e*n0/eps0*(Ni-Ne)
    p(x, V, E, Ni, Ne, Ui, Ue) = -(e*Ni*E/(gammai * kTi)/(Ui^2-1)) + Ne*nuiz/sqrt(gammai * kTi / mi)/Ui*(1+1/(Ui^2-1)) + nuim * ...
    h(x, V, E, Ni, Ne, Ui, Ue) = (e*Ne*E/(gammae * kTe)/(Ue^2-1)) + Ne*nuiz/sqrt(gammae * kTe / me)/Ue*(1+1/(Ue^2-1)) + nuem * ...
    m(x, V, E, Ni, Ne, Ui, Ue) = e*Ui*E/(gammai * kTi)/(Ui^2-1) - Ne*nuiz/sqrt(gammai * kTi / mi)/Ni*(1+1/(Ui^2-1)) - nuim * ...
    f(x, V, E, Ni, Ne, Ui, Ue) = -e*Ue*E/(gammae * kTe)/(Ue^2-1) - nuiz/sqrt(gammae * kTe / me)*(1+1/(Ue^2-1)) - nuem * ...
        

    """
    V = np.zeros(Nx)
    E = np.zeros(Nx)
    Ni = np.zeros(Nx)
    Ne = np.zeros(Nx)
    Ui = np.zeros(Nx)
    Ue = np.zeros(Nx)

    pcheck1 = np.zeros(Nx)
    pcheck2 = np.zeros(Nx)
    lcheck1 = np.zeros(Nx)
    lcheck2 = np.zeros(Nx)

    # Psi[0] = -0.5
    # Delta[0] = 50000
    # Ni[0] = m.exp(Psi[0])
    # Ne[0] = m.exp(Psi[0])
    V[0] = 0  # adjusted value
    E[0] = 0  # adjusted value
    Ni[0] = m.exp(V[0])
    Ne[0] = m.exp(V[0])
    # Ui[0] = 1.001
    Ui[0] = 6.25  # # adjusted value
    Ue[0] = 0.0031

    print(Ni[0])
    Uith = m.sqrt(gammai * kTi / mi)
    Ueth = m.sqrt(gammae * kTe / me)
    # nui=nue=0
    i = 0

    while (V[i] > Vl) and (i < Nx - 1):
        # print(i)
        k1 = dx * (-E[i])
        l1 = dx * e * n0 / eps0 * (Ni[i] - Ne[i])
        p1 = dx * (-(e*Ni[i]*E[i]/(gammai * kTi)/(Ui[i]*Ui[i]-1)) + Ne[i]*nuiz/Uith/Ui[i]*(1+1/(Ui[i]*Ui[i]-1)))
        h1 = dx * ((e*Ne[i]*E[i]/(gammae * kTe)/(Ue[i]*Ue[i]-1)) + Ne[i]*nuiz/Ueth/Ue[i]*(1+1/(Ue[i]*Ue[i]-1)))
        m1 = dx * ((e*Ui[i]*E[i]/(gammai * kTi)/(Ui[i]*Ui[i]-1)) - Ne[i]*nuiz/Uith/Ni[i]*(1+1/(Ui[i]*Ui[i]-1)))
        f1 = dx * (-(e*Ue[i]*E[i]/(gammae * kTe)/(Ue[i]*Ue[i]-1)) - nuiz/Ueth*(1+1/(Ue[i]*Ue[i]-1)))


        k2 = dx * (-E[i] - l1 / 2)
        l2 = dx * e * n0 / eps0 * (Ni[i] + p1 / 2 - Ne[i] - h1 / 2)
        p2 = dx * (-(e*(Ni[i]+p1/2)*(E[i]+l1/2)/(gammai * kTi)/((Ui[i]+m1/2)*(Ui[i]+m1/2)-1)) + (Ne[i]+h1/2)*nuiz/Uith/(Ui[i]+m1/2)*(1+1/((Ui[i]+m1/2)*(Ui[i]+m1/2)-1)))
        h2 = dx * ((e * (Ne[i] + h1 / 2) * (E[i] + l1 / 2) / (gammae * kTe) / (
                    (Ue[i] + f1 / 2) * (Ue[i] + f1 / 2) - 1)) + (Ne[i] + h1 / 2) * nuiz / Ueth / (Ue[i] + f1 / 2) * (
                               1 + 1 / ((Ue[i] + f1 / 2) * (Ue[i] + f1 / 2) - 1)))
        m2 = dx * ((e*(Ui[i]+m1/2)*(E[i] + l1 / 2)/(gammai * kTi)/((Ui[i]+m1/2)*(Ui[i]+m1/2)-1)) - (Ne[i]+h1/2)*nuiz/Uith/(Ni[i]+p1/2)*(1+1/((Ui[i]+m1/2)*(Ui[i]+m1/2)-1)))
        f2 = dx * (-(e*(Ue[i]+f1/2)*(E[i] + l1 / 2)/(gammae * kTe)/((Ue[i]+f1/2)*(Ue[i]+f1/2)-1)) - nuiz/Ueth*(1+1/((Ue[i]+f1/2)*(Ue[i]+f1/2)-1)))


        k3 = dx * (-E[i] - l2 / 2)
        l3 = dx * e * n0 / eps0 * (Ni[i] + p2 / 2 - Ne[i] - h2 / 2)
        p3 = dx * (-(e*(Ni[i]+p2/2)*(E[i]+l2/2)/(gammai * kTi)/((Ui[i]+m2/2)*(Ui[i]+m2/2)-1)) + (Ne[i]+h2/2)*nuiz/Uith/(Ui[i]+m2/2)*(1+1/((Ui[i]+m2/2)*(Ui[i]+m2/2)-1)))
        h3 = dx * ((e * (Ne[i] + h2 / 2) * (E[i] + l2 / 2) / (gammae * kTe) / (
                (Ue[i] + f2 / 2) * (Ue[i] + f2 / 2) - 1)) + (Ne[i] + h2 / 2) * nuiz / Ueth / (Ue[i] + f2 / 2) * (
                           1 + 1 / ((Ue[i] + f2 / 2) * (Ue[i] + f2 / 2) - 1)))
        m3 = dx * ((e * (Ui[i] + m2 / 2) * (E[i] + l2 / 2) / (gammai * kTi) / (
                    (Ui[i] + m2 / 2) * (Ui[i] + m2 / 2) - 1)) - (Ne[i] + h2 / 2) * nuiz / Uith / (Ni[i] + p2 / 2) * (
                               1 + 1 / ((Ui[i] + m2 / 2) * (Ui[i] + m2 / 2) - 1)))
        f3 = dx * (-(e * (Ue[i] + f2 / 2) * (E[i] + l2 / 2) / (gammae * kTe) / (
                    (Ue[i] + f2 / 2) * (Ue[i] + f2 / 2) - 1)) - nuiz / Ueth * (
                               1 + 1 / ((Ue[i] + f2 / 2) * (Ue[i] + f2 / 2) - 1)))


        k4 = dx * (-E[i] - l3)
        l4 = dx * e * n0 / eps0 * (Ni[i] + p3 - Ne[i] - h3)
        p4 = dx * (-(e*(Ni[i]+p3)*(E[i]+l3)/(gammai * kTi)/((Ui[i]+m3)*(Ui[i]+m3)-1)) + (Ne[i]+h3)*nuiz/Uith/(Ui[i]+m3)*(1+1/((Ui[i]+m3)*(Ui[i]+m3)-1)))
        h4 = dx * ((e * (Ne[i] + h3) * (E[i] + l3) / (gammae * kTe) / (
                (Ue[i] + f3) * (Ue[i] + f3) - 1)) + (Ne[i] + h3) * nuiz / Ueth / (Ue[i] + f3) * (
                           1 + 1 / ((Ue[i] + f3) * (Ue[i] + f3) - 1)))
        m4 = dx * ((e * (Ui[i] + m3) * (E[i] + l3) / (gammai * kTi) / (
                (Ui[i] + m3) * (Ui[i] + m3) - 1)) - (Ne[i] + h3) * nuiz / Uith / (Ni[i] + p3) * (
                           1 + 1 / ((Ui[i] + m3) * (Ui[i] + m3) - 1)))
        f4 = dx * (-(e * (Ue[i] + f3) * (E[i] + l3) / (gammae * kTe) / (
                (Ue[i] + f3) * (Ue[i] + f3) - 1)) - nuiz / Ueth * (
                           1 + 1 / ((Ue[i] + f3) * (Ue[i] + f3) - 1)))

        # pcheck1[i] = kTe*Delta[i]*Ni[i]-m.sqrt(mi*kTi)*nu
        # pcheck2[i] = gammai * m.pow(Ni[i], gammai+1) - 1
        # lcheck1[i] = Ni[i]
        # lcheck2[i] = m.exp(Psi[i])
        # print(p1)
        # print(B * quad(FN, Psi0, Psi[i]+ dx * f3)[0])
        V[i + 1] = V[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        E[i + 1] = E[i] + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        Ni[i + 1] = Ni[i] + 1 / 6 * (p1 + 2 * p2 + 2 * p3 + p4)
        Ne[i + 1] = Ne[i] + 1 / 6 * (h1 + 2 * h2 + 2 * h3 + h4)
        Ui[i + 1] = Ui[i] + 1 / 6 * (m1 + 2 * m2 + 2 * m3 + m4)
        Ue[i+1] = Ue[i] + 1 / 6 * (f1 + 2 * f2 + 2 * f3 + f4)

        i = i + 1
    """
    plt.plot(pcheck1, 'b')
    plt.ylabel('p chisl')
    # plt.ylabel('Ni')
    plt.show()

    plt.plot(pcheck2, 'r')
    plt.ylabel('p znam')
    # plt.ylabel('expPsi')
    plt.show()
    """
    Nel = i + 1

    return V, E, Ni, Ne, Ui, Ue, Nel


def main():
    # initialisation of parameters
    boxsize = 1.5E-3  # m
    dx = 1E-6
    Nx = int(boxsize / dx)
    Nsh = 0

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.68  # eV
    Ti = 0.05  # eV
    n0 = 3E17  # m-3
    Vdc = -17
    gammai = 1
    gammae = 1
    # nu = 4e8
    nui = 0
    # nue = 4e12
    nue = 0
    nuiz = 1e5  # adjusted value
    # nuiz = 0

    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J

    x = np.arange(Nx) * dx
    V = np.zeros(Nx)
    ni = np.zeros(Nx)
    ne = np.zeros(Nx)
    ui = np.zeros(Nx)
    ue = np.zeros(Nx)
    ji = np.zeros(Nx)
    je = np.zeros(Nx)

    Psi = np.zeros(Nx)
    Ni = np.zeros(Nx)
    Ui = np.zeros(Nx)
    Delta = np.zeros(Nx)

    Vl = Vdc
    #Psi0 = -1e-7


    V, E, Ni, Ne, Ui, Ue, Nel = RungeKuttasystem(Nx, dx, n0, Te, Ti, Vl, gammai, gammae, nui, nue, nuiz)

    for i in range(0, Nel):
        #V[i] = Psi[i] * kTe / e
        ni[i] = Ni[i] * n0
        ne[i] = Ne[i]*n0
        #ne[i] = n0 * m.exp(e * V[i] / kTe)
        ui[i] = Ui[i] * m.sqrt(gammai * kTi / mi)
        ue[i] = Ue[i] * m.sqrt(gammae * kTe / me)
        ji[i] = ni[i] * ui[i]
        je[i] = ne[i] * ue[i]
        # ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]
        # ue[i] = n0 * m.sqrt(kTe / me) / ne[i]
    """
    ne[0] = n0
    for i in range(1, Nel-1):
        ne[i] = ni[i] + eps0 / e * (V[i-1] + 2 * V[i] - V[i+1]) / dx / dx
    """
    """
    ni[0] = n0
    ui[0] = n0 * m.sqrt(kTi / mi) / ni[0]
    for i in range(1, Nel - 1):
        ni[i] = ne[i] - eps0 / e * (V[i - 1] + 2 * V[i] - V[i + 1]) / dx / dx
        ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]
    """
    # print(Psi)

    plt.plot(x, Psi)
    plt.ylabel('Psi')
    plt.show()
    """
    plt.plot(x, Ne, 'b')
    plt.plot(x, Ni, 'r')
    plt.ylabel('Ni')
    plt.show()
    """
    plt.plot(x, Delta)
    plt.ylabel('-dPsi/dx')
    plt.show()

    plt.plot(x, V)
    plt.ylabel('V')
    plt.xlabel('x')
    plt.show()

    plt.plot(x, ne, 'b')
    plt.plot(x, ni, 'r')
    plt.ylabel('N')
    plt.xlabel('x')
    plt.show()

    # plt.plot(x, ue, 'b')
    plt.plot(x, ui, 'r')
    plt.plot(x, ue, 'b')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.show()

    plt.plot(x, ji, 'r')
    plt.plot(x, je, 'b')
    plt.ylabel('j')
    plt.xlabel('x')
    plt.show()

    f = open("V.txt", "w")
    for d in V:
        f.write(f"{d}\n")
    f.close()

    f = open("ni.txt", "w")
    for d in ni:
        f.write(f"{d}\n")
    f.close()

    f = open("ne.txt", "w")
    for d in ne:
        f.write(f"{d}\n")
    f.close()

    f = open("ui.txt", "w")
    for d in ui:
        f.write(f"{d}\n")
    f.close()

    f = open("ue.txt", "w")
    for d in ue:
        f.write(f"{d}\n")
    f.close()

    f = open("Nel.txt", "w")
    f.write(f"{Nel}\n")
    f.close()

    return 0


if __name__ == "__main__":
    main()