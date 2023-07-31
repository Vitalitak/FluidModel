import math as m
import matplotlib.pyplot as plt
import numpy as np
import time

def main():

    boxsize = 5E-3  # m
    dx = 1E-5

    N = int(boxsize/dx)
    x = [k * dx for k in range(0, N)]
    x = np.array(x)

    with open("Vt.txt", "r") as f:
        lines = f.readlines()
        #print(int(len(lines)/N))
        Vt = np.zeros((int(len(lines) / N), N))

        for i in range(0, int(len(lines)/N)):
            k = 0
            for j in range(int(i*N), int((i+1)*N)):
                Vt[i, k] = float(lines[j])
                k += 1

    f.close()

    i = 0
    V = [0 for k in range(0, N)]

    with open("V.txt", "r") as f1:
        for line in f1.readlines():
            for ind in line.split():
                V[i] = float(ind)
                i += 1
    f1.close()

    V = np.array(V)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.xlabel('x')
    plt.ylabel('V')

    for i in range(0, int(len(lines)/N) - 1):
        plt.plot(x, Vt[i, ], 'r-')
        plt.plot(x, V, 'r--')
        plt.ylim(-40, 2)
        plt.grid(axis = 'y')
        plt.pause(0.15)
        plt.cla()

        #time.sleep(0.)
    #plt.ylabel('V')
    #plt.xlabel('x')
    plt.plot(x, Vt[int(len(lines)/N)-1,], 'r-')
    plt.plot(x, V, 'r--')
    plt.ylim(-40, 2)
    plt.grid(axis='y')
    plt.show()

    """
    plt.plot(x, Vt[0, ], 'r')
    plt.pause(0.5)
    """
    """
    plt.plot(x, Vt[5, ], 'b')
    plt.plot(x, Vt[9, ], 'm')
    plt.plot(x, Vt[14, ], 'g')
    plt.plot(x, Vt[55, ], 'r--')

    plt.ylabel('V')
    plt.xlabel('x')
    plt.show()
    """
    return 0

if __name__ == "__main__":
    main()