import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, sqrt
from imageio import imread
from math import floor, ceil

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

def coeff(p1, p2): #der p1 er punkte på portrett og p2 på maske
    x1, y1, x2, y2 , ones = [], [], [], [], [1,1,1]
    matrix_portrett = [y1, x1, ones]
    matrix_maske = [y2, x2, ones]
    for i in range(len(p1)):
        a, b = p1[i] 
        c, d = p2[i] 
        x1.append(a)
        y1.append(b)
        x2.append(c)
        y2.append(d)
    co = (matrix_maske) @ np.linalg.inv(matrix_portrett)
    #print('coefficient matrix:', co)
    return co


def T(x,y, coefficient):
    A = [x, y, 1]
    return coefficient@A


def affine(bilde, mapping, p1, p2): #samregistrering
    a, b = mapping.shape #fra bilde vi skal mappe til
    N, M = bilde.shape #potrett bilde
    out = np.zeros((a,b)) #512x600
    c = coeff(p1,p2) #beregner koeffisient matrisen
    for x in range(N):
        for y in range(M):
            x_new, y_new, n = T(x,y,c)
            if int(y_new) < 512 and int(x_new )< 600 and y_new >= 0 and x_new>=0:
                out[int(x_new), int(y_new)] = bilde[x, y]
    return out

p1 = [[84,88], [120,67], [129, 108]] #punkter fra potrett (øyner og munn)
p2 = [[170,259], [342,259], [256,441]]#punkter på maske vi skal mappe til


#oppgave 2 
def bilin(f_in, f_portrett, p1, p2): #bilinear interpolasjon, koden funker ikke, har ikke fullført oppgave
    N, M = f_in.shape #potrett bilde som ble transformert
    f_ut = np.zeros((N,M)) 
    c_inv = np.linalg.inv(coeff(p1,p2))
    for i in range (N):
        for j in range (M):
            x, y, n = T(i,j, c_inv)
            x0, y0 = floor(x), floor(y)
            x1, y1 = ceil(x), ceil(y)
            dx, dy = x-x0, y-y0
            p = f_portrett[x0,y0] + (f_portrett[x1,y0] - f_portrett[x0, y0])*dx
            q = f_portrett[x0,y1] + (f_portrett[x1,y1] - f_portrett[x0, y1])*dx
            f_ut[i,j] = p+(q-p)*dy
    return  f_ut

def nabo(f_in, f_portrett, p1, p2):
    M,N = f_in.shape
    final_pic = np.zeros((M,N))
    c_inv = np.linalg.inv(coeff(p1,p2))
    for i in range(M):
        for j in range(N):
            x, y, n = T(i,j, c_inv)
            x, y = int(x), int(y)
            final_pic[i,j] = f_portrett[x,y]
    return final_pic

img = imread('portrett.png', as_gray=True)
maske = imread('geometrimaske.png', as_gray= True)

portrett_kont = lin_t(img, 64, 127) #linear gråtone transformation, tar in sigma og 
img_n = affine(portrett_kont, maske, p1, p2)
im_bilin = bilin(img_n, portrett_kont, p1, p2)
im_nabo = nabo(img_n, portrett_kont, p1, p2)

print('middelverdi (original)', middel(img))
#print(np.mean(img)) #sjekker med ferdig pakke
print('sigma (original):', deviation(img))
#print(np.std(img)) #sjekker med ferdig pakke

print('sigma (new image):', deviation(img_n))
print('middelverdi (new image):', middel(img_n))


plt.figure()
plt.title("Original portrait")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

plt.figure()
plt.title("Gråtone transformert")
plt.imshow(portrett_kont, cmap='gray', vmin=0, vmax=255)
plt.show()

plt.figure()
plt.title("Mapped")
plt.imshow(img_n, cmap= 'gray', vmin=0, vmax=255)
plt.show()

plt.figure()
plt.title("Bilinær interpolasjon")
plt.imshow(im_bilin, cmap= 'gray', vmin=0, vmax=255)

plt.figure()
plt.title("Nabo")
plt.imshow(im_nabo, cmap= 'gray', vmin=0, vmax=255)
plt.show()



