from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin

f = imread('mona.png', as_gray=True)


def transformation(f_in, transform):
    N,M = f_in.shape
    f_out = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            vec_in = np.array([i, j, 1])
            vec_out = np.dot(transform, vec_in)
            x = int(vec_out[0])
            y = int(vec_out[1])
            if (x in range(N) and y in range(M)):
                f_out[x, y] = f_in[i, j]
    return f_out

translate = np.array([[1, 0, len(f)//2],
                      [0, 1, len(f)//2],
                      [0, 0, 1]])

degree = 33
th =  degree*pi/180

rotate = np.array([[cos(th), -sin(th), 0],
                   [sin(th), cos(th), 0],
                   [0, 0, 1]])

scale = np.array([[0.5, 0, 0],
                  [0, 0.5, 0],
                  [0, 0, 1]])

new = rotate.dot(translate.dot(scale))

g = transformation(f, new)


plt.figure()
plt.title("Translated")
plt.imshow(g, cmap='gray')

plt.show()
