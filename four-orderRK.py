import math as m
import matplotlib.pyplot as plt
import numpy as np

"""
four order Runge-Kutta method for solution equation
dy/dx=f(x, y)

"""
def RKPois(h, y0, Nx)

    y = [0 for k in range(0, Nx)]

    y[0] = y0



    return y