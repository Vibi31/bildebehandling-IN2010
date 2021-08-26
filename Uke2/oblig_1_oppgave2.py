import matplotlib.pyplot as plt
import numpy as np
from numpy import rot90, arctan, sqrt, pi, exp
from math import ceil, floor
from imageio import imread

def konvolusjon(img, fil): #tar inn-bilde og filteren 
    M, N = img.shape #inn bilde
    konvulert = np.zeros((M,N)) #lager ut-bilde like stor som innbilde
    x, y = fil.shape #filter size 

    #Utvider med nærmeste pikselverdi (løser bilderandproblemet)
    a, b = int(0.5*(x-1)), int(0.5*(y-1))  #rader og kolonner vi må nullutvide 
    pad_img = np.pad(img, ((a,a),(b,b)), mode='edge') 

    #rotere filteren med 180 grader (ikke nødvendig om filteret er symetrisk)
    rot_filter = rot90(fil, 2) 
    for i in range(M):
        for j in range(N):
            # Ganger overlappende verdier med rotert-filter og gir summen av de
            konvulert[i,j] = np.sum(pad_img[i:(i+ x), j:(j+ y)]*rot_filter)

    return konvulert

def canny(img, Tl, Th, sigma):

    #Lavpassfiltrer med Gauss-filter (med gitt sigma)
    filt = gauss(sigma)
    filt_img = konvolusjon(img, filt)

    #Finn gradient-magnituden og gradient-retningen.
    hx = [[0,1,0],[0,0,0],[0,-1,0]]  #symmetrisk 1D operator fra slide 25: 
    hy = [[0,0,0],[1,0,-1],[0,0,0]]  #symmetrisk 1D operator fra slide 25: 
    gx = konvolusjon(filt_img, hx)
    gy = konvolusjon(filt_img, hy)
    magnitude = sqrt(gx**2 + gy**2)
    angle = np.rad2deg(arctan(gy,gx)) 

    #Tynning av gradient-magnitude ortogonalt på kant. har ikke fullført
    #Hysterese-terskling (to terskler, Th og Tl), har ikke fullført


def gauss(sigma):                           #gauss_filter, tar in sigma
    dim = int(8*sigma + 1)
    g_dim = 4*sigma                         #halv parten of gauss
    g_filter = np.zeros((dim,dim))          #lager tom gauss filter
    norm = 1 / (2* pi * sigma**2)

    for x in range(-g_dim , g_dim+1):       #sånn at 0 er midtpunkten
        for y in range(-g_dim , g_dim+1):
            g_filter[x+g_dim, y+g_dim] = exp(-((x**2 + y**2) / (2*sigma**2)))
    g = g_filter * norm                     # normaliserer filteret
    return g



b = np.array([ #matrise fra slide 40 -forelesning
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]])

img = imread('cellekjerner.png', as_gray=True)
i = konvolusjon(img, b)

plt.figure()
plt.title("cellekjerner")
plt.imshow(img, cmap='gray')

plt.figure()
plt.title("konvulert med matris b fra forelesning slide 40")
plt.imshow(i, cmap='gray')

sigma4 = konvolusjon(img, gauss(4))

plt.figure()
plt.title("konvulert med sigma = 4")
plt.imshow(sigma4, cmap='gray')
plt.show()