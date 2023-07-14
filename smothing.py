import math as m
import matplotlib.pyplot as plt
import numpy as np

def main():
    # initialisation of parameters
    boxsize = 5E-3  # m
    dx = 1E-6
    Nx = int(boxsize / dx)

    # read initial conditions from file

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]

    i = 0
    with open("V.txt", "r") as f1:
        for line in f1.readlines():
            for ind in line.split():
                V[i] = float(ind)
                i += 1
    f1.close()
    i = 0
    with open("ni.txt", "r") as f2:
        for line in f2.readlines():
            for ind in line.split():
                ni[i] = float(ind)
                i += 1
    f2.close()
    i = 0

    with open("ne.txt", "r") as f3:
        for line in f3.readlines():
            for ind in line.split():
                ne[i] = float(ind)
                i += 1
    f3.close()
    i = 0

    with open("ui.txt", "r") as f4:
        for line in f4.readlines():
            for ind in line.split():
                ui[i] = float(ind)
                i += 1
    f4.close()
    i = 0

    with open("ue.txt", "r") as f5:
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

    x = np.array(x)
    V = np.array(V)
    ni = np.array(ni)
    ne = np.array(ne)
    ui = np.array(ui)
    ue = np.array(ue)

    # smoothing

    f = open("V_sm.txt", "w")
    for d in V:
        f.write(f"{d}\n")
    f.close()

    f = open("ne_sm.txt", "w")
    for d in ne:
        f.write(f"{d}\n")
    f.close()

    f = open("ni_sm.txt", "w")
    for d in ni:
        f.write(f"{d}\n")
    f.close()

    f = open("ue_sm.txt", "w")
    for d in ue:
        f.write(f"{d}\n")
    f.close()

    f = open("ui_sm.txt", "w")
    for d in ui:
        f.write(f"{d}\n")
    f.close()


    return 0


if __name__ == "__main__":
    main()