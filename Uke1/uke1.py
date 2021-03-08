#bildebehandling
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np 

#del 1
filename = 'mona.png'
f = imread(filename, as_gray=True) #, flatten=True

#N is number of rows, M = column
N,M = f.shape
f_out = np.zeros((N,M)) #empty array with space for all pixels

f_out[0,:] = f[0,:] #setting first row of out-picture equal to the in-picture

for i in range(1,N):
  for j in range(M):
    f_out[i,j] = f[i,j] - f[i-1, j]

bias = 128 #to get rid of negative values that represent black
plt.figure()
plt.title("contrast=1 bias=128")
plt.imshow(f_out + bias,cmap='gray',vmin=0,vmax=255)

plt.figure()
plt.title("contrast=1 bias=0")
plt.imshow(f_out ,cmap='gray',vmin=0,vmax=255)

plt.figure()
plt.title("contrast=2, bias=128")
plt.imshow(2*f_out + bias, cmap ='gray', vmin=0, vmax=255)

plt.figure()
plt.title("contrast=2, bias=0")
plt.imshow(f_out*2,cmap='gray',vmin=np.min(f_out), vmax=np.max(f_out))

plt.show()
#with higher contrast we get a more 3D type effect, easier to tell the difference between neighbouring pixels

"""
# Alternativ andre del med litt mindre kodelinjer som gjør akkurat det samme:
f_out_alt = 0.5*f
plt.figure()
plt.imshow(f_out_alt,cmap='gray',vmin=0,vmax=255,aspect='auto')
plt.title('f_out_alt')
"""
plt.show()