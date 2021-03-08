#bildebehandling
from imageio import imread
import matplotlib.pyplot as pyplot
import numpy as np 

f = imread('mona.png', as_gray = true)
print(f)

N,M = f.shape
f_out = np.zeros((N,M))
print(_out)

plt.figure()
plt.imshow(f, cmap = 'gray', vmin = 0, vmax =255)
plt.title('Mona')
plt.show()