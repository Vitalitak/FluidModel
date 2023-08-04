import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def main():
    # initialisation of parameters
    boxsize = 5E-3  # m
    dx = 1E-5
    Nx = int(boxsize / dx)
    N = 7

    # read initial conditions from file

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]
    ni_0 = [0 for k in range(0, Nx)]
    ne_0 = [0 for k in range(0, Nx)]

    i = 0
    with open("V_1.txt", "r") as f1:
        for line in f1.readlines():
            for ind in line.split():
                V[i] = float(ind)
                i += 1
    f1.close()
    i = 0
    with open("ni_1.txt", "r") as f2:
        for line in f2.readlines():
            for ind in line.split():
                ni[i] = float(ind)
                i += 1
    f2.close()
    i = 0

    with open("ne_1.txt", "r") as f3:
        for line in f3.readlines():
            for ind in line.split():
                ne[i] = float(ind)
                i += 1
    f3.close()
    i = 0

    with open("ui_1.txt", "r") as f4:
        for line in f4.readlines():
            for ind in line.split():
                ui[i] = float(ind)
                i += 1
    f4.close()
    i = 0

    with open("ue_1.txt", "r") as f5:
        for line in f5.readlines():
            for ind in line.split():
                ue[i] = float(ind)
                i += 1
    f5.close()
    i = 0

    with open("Nel.txt", "r") as f6:
        for line in f6.readlines():
            Nel = int(line)
    f6.close()

    with open("ni.txt", "r") as f7:
        for line in f7.readlines():
            for ind in line.split():
                ni_0[i] = float(ind)
                i += 1
    f7.close()
    i = 0

    with open("ne.txt", "r") as f8:
        for line in f8.readlines():
            for ind in line.split():
                ne_0[i] = float(ind)
                i += 1
    f8.close()
    i = 0

    x = np.array(x)
    V = np.array(V)
    ni = np.array(ni)
    ne = np.array(ne)
    ui = np.array(ui)
    ue = np.array(ue)
    ni_0 = np.array(ni_0)
    ne_0 = np.array(ne_0)

    V_sm = np.zeros(Nx)
    ne_sm = np.zeros(Nx)
    ni_sm = np.zeros(Nx)
    ue_sm = np.zeros(Nx)
    ui_sm = np.zeros(Nx)

    # smoothing
    """
    for i in range(0, Nel -N +1):
        for j in range(0, N):
            V_sm[i] += V[i+j]
            ne_sm[i] += ne[i+j]
            ni_sm[i] += ni[i+j]
            ue_sm[i] += ue[i+j]
            ui_sm[i] += ui[i+j]
        V_sm[i] /= N
        ne_sm[i] /= N
        ni_sm[i] /= N
        ue_sm[i] /= N
        ui_sm[i] /= N
    """
    #Vpre = np.zeros(Nel)
    ne_pre = np.zeros(Nel)
    ni_pre = np.zeros(Nel)
    #ue_pre = np.zeros(Nel)
    #ui_pre = np.zeros(Nel)
    #Vpre[0:Nel] = V[0:Nel]
    ne_pre[0:Nel] = ne[0:Nel]
    ni_pre[0:Nel] = ni[0:Nel]
    #ue_pre[0:Nel] = ue[0:Nel]
    #ui_pre[0:Nel] = ui[0:Nel]

    #V_sm[0:Nel] = signal.savgol_filter(Vpre, window_length=N, polyorder=3)
    ne_sm[0:Nel] = signal.savgol_filter(ne_pre, window_length=N, polyorder=3)
    ni_sm[0:Nel] = signal.savgol_filter(ni_pre, window_length=N, polyorder=3)
    #ue_sm[0:Nel] = signal.savgol_filter(ue_pre, window_length=N, polyorder=3)
    #ui_sm[0:Nel] = signal.savgol_filter(ui_pre, window_length=N, polyorder=3)

    V_sm[0:Nx] = V[0:Nx]
    #ne_sm[0:Nx] = ne[0:Nx]
    #ni_sm[0:Nx] = ni[0:Nx]
    ne_sm[0] = ne[0]
    ni_sm[0] = ni[0]
    ne_sm[Nel-4:Nel] = ne[Nel-4:Nel]
    ni_sm[Nel-1] = ni[Nel-1]
    ui_sm[0:Nx] = ui[0:Nx]
    ue_sm[0:Nx] = ue[0:Nx]

    plt.plot(x, V, 'r--')
    plt.plot(x, V_sm, 'r-')
    plt.ylabel('V')
    plt.show()

    plt.plot(x, ni, 'r--')
    plt.plot(x, ni_sm, 'r-')
    #plt.plot(x, ni_0, 'm')
    plt.plot(x, ne, 'b--')
    #plt.plot(x, ne_0, 'b-')
    plt.plot(x, ne_sm, 'b-')
    # plt.plot(x, ni_2, 'g')
    # plt.plot(x, ni_3, 'm')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, ui, 'r--')
    plt.plot(x, ui_sm, 'r-')
    plt.plot(x, ue, 'b--')
    plt.plot(x, ue_sm, 'b-')
    plt.ylabel('u')
    plt.show()

    f = open("V_sm.txt", "w")
    for d in V_sm:
        f.write(f"{d}\n")
    f.close()

    f = open("ne_sm.txt", "w")
    for d in ne_sm:
        f.write(f"{d}\n")
    f.close()

    f = open("ni_sm.txt", "w")
    for d in ni_sm:
        f.write(f"{d}\n")
    f.close()

    f = open("ue_sm.txt", "w")
    for d in ue_sm:
        f.write(f"{d}\n")
    f.close()

    f = open("ui_sm.txt", "w")
    for d in ui_sm:
        f.write(f"{d}\n")
    f.close()


    return 0


if __name__ == "__main__":
    main()