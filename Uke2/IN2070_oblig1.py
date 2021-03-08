from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, mean, sqrt

img = imread('portrett.png', as_gray=True)
N, M = img.shape
pix = N*M

def middel(img):         #calculates mean value without using libraries        
    verdi_sum = 0
    for i in range (N):
        for j in range (M):
            verdi_sum += img[i,j]
    return verdi_sum/pix

def deviation(img): #calculates deviation without using libraries
    mx = middel(img)
    tot = 0
    for i in range (N):
        for j in range (M):
            tot += (img[i,j] - mx)**2
    return sqrt(tot/pix)

def lin_mean(img, m1): #tar in bilde og ønsket middel verdi
    f_mean = np.zeros((N,M))
    m0 = middel(img)
    for i in range(N):
        for j in range(M):
            f_mean[i, j] = img[i, j] + (m1 - m0)
    return f_mean

def lin_std(img, std1):
    std0 = deviation(img)
    C = std1/std0
    f_std = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_std[i, j] = img[i, j]*C
    return f_std

def lin_t(img, std1, mv1): #tar in bilde, ønsket sigma og middelverdi
    std0, mv0 = deviation(img), middel(img)
    C = std1/std0
    f_lin = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_lin[i, j] = (img[i,j] - mv0) * (std1/std0) +  mv1
    return f_lin

img_n = lin_t(img, 64, 127)
img_b = lin_mean(img, 1800)
print(img)
print('middelverdi (original)', middel(img))
#print(np.mean(img)) #sjekker med ferdig pakke
print('sigma (original):', deviation(img))
#print(np.std(img)) #sjekker med ferdig pakke

print('sigma (new image):', deviation(img_n))
print('middelverdi (new image):', middel(img_n))

plt.figure()
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.figure()
plt.title("gråtone transform")
plt.imshow(img_n, cmap= 'gray', vmin=0, vmax=255)
plt.show()
