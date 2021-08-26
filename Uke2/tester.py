import matplotlib.pyplot as plt
from scipy import signal
from imageio import imread
import numpy as np
from numpy import cos, pi, sqrt

#steg 1: laster inn bilde med spesifisert parameter
img = imread('uio.png', as_gray=True)

def C(num):
    if num == 0:
        return 1/sqrt(2)
    else:
        return 1

def transform(f):
    F = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            left = (1/4)*C(u)*C(v)
            right = 0
            for y in range(8):
                for x in range(8):
                    right += f[y][x]*cos(((2*x+1)*u*pi)/16)*cos(((2*y+1)*v*pi)/16)
            F[u][v] = left*right
    return F


transform(img)