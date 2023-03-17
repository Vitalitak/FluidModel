import math as m
import matplotlib.pyplot as plt
import numpy as np

def plotpotential(Te, Ti, Vp):

    Nl = int(pow(-2*Vp/5/Ti, 1.5))
    N0 = 1

    n = [k for k in range(N0, Nl)]
    phi = [0 for k in range(N0, Nl)]
    phi1 = [0 for k in range(N0, Nl)]
    phi2 = [0 for k in range(N0, Nl)]


    for i in range(N0-1, Nl-1):
        print(i)
        phi[i] = 3*Ti/Te*(1-5/6*pow(n[i], 0.66)-1/6/pow(n[i], 2))
        phi1[i] = -2.5*Ti/Te*pow(n[i], 0.66)
        phi2[i] = 0.5*Ti / Te * (1 - pow(n[i], -2))

    plt.plot(phi, 'b', phi1, 'r', phi2, 'g')
    plt.axis([N0-10, Nl+10, -10, 0.1])
    plt.ylabel('phi')
    plt.text(500, -2, r'blue - solution, red, green - aprox')
    plt.show()

    return 0

def main():

    Te = 2.3
    Ti = 0.06
    Vp = -18

    plotpotential(Te, Ti, Vp)
    phiL = Vp/Te
    print(phiL)

    return 0

if __name__ == "__main__":
    main()
