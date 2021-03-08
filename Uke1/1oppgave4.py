from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

f = imread('mona.png',as_gray=True)

for bit in range(1,8,2):

    quantized = f//(2**(8-bit)) #whole nuber division.
    plt.figure() 
    plt.title("bit = %d, antall verdier = %d"%(bit, np.max(quantized) + 1))
    plt.imshow(quantized,cmap='gray')

plt.show()

#oppgave5
hd = 1080*1920 #total pixels
bit = 3 #bit per pixel, r,g,b
fps = 50 #number of frames per second
bit = 8+8+8 #r=8, g=8, b=8
byte = bit/8
time = 2*60*60 #two hours, in seconds

print('bytes =', hd*fps*byte*time)
print('gigabyte =',(hd*fps*byte*time)/1000000000)

#oppgave 6
#Siden like mange forgrunns- som bakgrunnspiksler,
# ville kvantiseringsterskelen bli lagt midt mellom 50 og 200, 
# (50+200)/2 = 125. Rekonstruksjon hvor 0 ble gjort om til 50 og 1 til 200.