from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, mean, sqrt


img = imread('portrett.png', as_gray=True)
m1 = 127  #ønsket middelverdi
b = m1 - mean(img) #økning for gråtone til hver piksel
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
    f_out = np.zeros((N,M))
    m0 = middel(img)
    for i in range(N):
        for j in range(M):
            f_out[i, j] = img[i, j] + (m1 - m0)
    return f_out

def lin_std(img, std1):
    std0 = deviation(img)
    C = std1/std0
    f_out = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_out[i, j] = img[i, j]*C
    return f_out

def lin_t(img, std1, mv1): #tar in bilde, ønsket sigma og middelverdi
    std0, m0 = deviation(img), middel(img)
    C = std1/std0
    N,M = img.shape
    f_out = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_out[i, j] = img[i, j] + (m1 - m0)*C
    return f_out


img_n = lin_t(img, 64, 127)


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
plt.title("linearly transformed (grey tone)")
plt.imshow(img_n, cmap='gray')

plt.show()
