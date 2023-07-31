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

    for i in range(0, int(len(lines)/N)):
        plt.plot(x, Vt[i, ], 'r')
        plt.ylim(-40, 2)
        plt.pause(0.1)
        plt.cla()

        #time.sleep(0.)
    #plt.ylabel('V')
    #plt.xlabel('x')
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