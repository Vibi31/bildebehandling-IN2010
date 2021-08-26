import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from imageio import imread

img = imread('cow.png', as_gray=True)
N, M = img.shape
pix = N*M


def filter(N):
    mv_filter = np.zeros((N,N))         #tom filter array med NxN størelse 
    for i in range(N):
        for j in range(N):
            mv_filter[i][j] = 1/(N**2)  #N^2 gir total elementer i filteret
    return mv_filter

#15x15 middelverdifiltrering gjennom romlig konvolusjon
mv_15 = filter(15)                                      #får en middelverdi 15x15 filter
img_15 = signal.convolve2d(img, mv_15, 'valid')         #python's versjon av matlab 'convolve2'


#15x15 middelverdifiltrering gjennom Fourier
filt_f = np.fft.rfft2(mv_15, s=(N, M))                  #real Fast Fourier Transform 2D (filter)
image_f = np.fft.rfft2(img)                             #real Fast Fourier Transform 2D (image)
img_f = np.fft.irfft2(image_f*filt_f)                   #invers real fourier transform (irfft)


plt.figure()
plt.title("Original")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

plt.figure()
plt.title("15x15 middelverdi-filtrert (konvulusjon)")
plt.imshow(img_15, cmap='gray', vmin=0, vmax=255)
plt.show()

plt.figure()
plt.title("15x15 middelverdi-filtrert (fourier)")
plt.imshow(img_f, cmap='gray', vmin=0, vmax=255)
plt.show()



#Oppgave 1.3
import time

def time_of(image, size):                          #tar in bilden og filter størelsen vi bruker
    mv_f = filter(size)                            #kjører funksjon som gir middelverdi filter-array
  
    #del 1, regner tiden på romlig konvulusjon
    start1 = time.time()                              
    img_conv = signal.convolve2d(img, mv_f, 'valid')
    romlig_conv_time =  time.time() - start1

    #del 2, regner tiden på FFT
    start2 = time.time()
    filt_f = np.fft.rfft2(mv_f, s=(N, M))         #real Fast Fourier Transform 2D (filter)
    image_f = np.fft.rfft2(img)                   #real Fast Fourier Transform 2D (image)
    img_f = np.fft.irfft2(image_f*filt_f)         #invers real fourier transform (irfft)
    fourier_time = time.time() - start2

    return(romlig_conv_time, fourier_time) 


img = imread('cow.png', as_gray=True)

n = np.linspace(5, 50, 10)     #array som har tallene som brukes for filter dimensjon
#print(n)                      #sjekker om jeg får rikktig array

konv = []
fourier = []
for i in range(len(n)):        #loop som skal teste tiden til dimensjonene i n-array
    size = int(n[i])
    k, f = time_of(img, size)
    konv.append(k)
    fourier.append(f)


                         
plt.plot(n, konv, label = 'konvulusjon')
plt.plot(n, fourier, label = 'fourier')
plt.title('Romlig konv. og Fourier Kjøretider')
plt.xlabel('Filter dimensjon [NxN]')
plt.ylabel('Tid [sekunder]')
plt.legend()
plt.show()



