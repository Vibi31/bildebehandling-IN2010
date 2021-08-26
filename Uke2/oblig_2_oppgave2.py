
import matplotlib.pyplot as plt
from scipy import signal
from imageio import imread
import numpy as np
from numpy import cos, pi, sqrt, log2, ravel

#steg 1: laster inn bilde med spesifisert parameter
img = imread('uio.png', as_gray=True)
N, M = img.shape
"""
print(np.max(img))                  #sjekker om max pixel verdi overstiger 255 (fikk 255)
print(N,M)
#sjekker om høyde og bredder er multiplum av 8:
print(N/8)                          #terminal: 31
print(M/8)                          #terminal: 62
"""

#steg 2: subtraher 128 fra pikselene
def steg2(img, subtract):
    for i in range(N):
        for j in range(M):
            img[i,j] = img[i,j] - subtract
    
    return img

img_128 = steg2(img, 128)

plt.figure()
plt.title("Original")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

print(np.max(img_128))
print(np.min(img_128))

plt.figure()
plt.title("steg 2")
plt.imshow(img_128, cmap='gray')
plt.show()


#steg 3: del bilde i 8x8 blokker
def C(num):
    if num == 0:
        return 1/sqrt(2)
    else:
        return 1


def DCT(img):                               #tar in en 8x8 bilde blokk, utfører DCT
    block = np.zeros((8,8))  
    for u in range(8):
        for v in range(8):
            tot = 0                         #tar summen til 0 før den regner summen igjen
            constant = 0.25 * C(u) *C(v)    #regner kostanten av DCT ligningen
            
            for x in range(8):              #regner sigma x og y av DCT ligningen
                for y in range(8):
                    tot += img[x][y] * cos((2*x+1)*u*pi/16) * cos((2*y+1)*v*pi/16) 

            block[u][v] = np.round(constant * tot)
    return block


def eight_block(img):                               #funksjonene tar in bilde og deler i blokk
    N, M = img.shape
    Nb, Mb = int(N/8) , int(M/8)                    #deler dimensjonen
    blocks = np.zeros((Nb, Mb), dtype = np.ndarray) #tom array som skal fylles med [8x8]blokker
    for i in range (Nb):
        for j in range (Mb):
            x, y = i*8, j*8 
            blocks[i][j] = DCT(img[x:x+8,  y:y+8])  #sender blokken til DCT-funksjon først

    return blocks
    #print(blocks)

blocks = eight_block(img_128)


#steg 4:Rekonstruer opprinnelige bilde
def IDCT(block):                               #regner invers DCT per blokk
    recon_block = np.zeros((8,8))
    for x in range(8):
        for y in range(8):
            piksel = 0
            for u in range(8):
                for v in range(8):           #IDCT 
                    piksel +=(
                        C(u)*C(v)
                        *(block[u][v])
                        *cos((2*x+1)*u*pi/16)
                        *cos((2*y+1)*v*pi/16)
                        )
            recon_block[x][y] = float(0.25*piksel + 128)   #addere piksel intensitet
    return recon_block                         #returnere invers block med riktig piksel verdi


def reconstruct(blocks):                     #regner invers DCT og setter blokkene sammen
    N, M = img.shape
    Nb, Mb = int(N/8) , int(M/8)
    recon = np.zeros((N,M), dtype=np.ndarray)

    for i in range (Nb):
        for j in range (Mb):
            x, y = i*8, j*8 
            recon[x:x+8,  y:y+8] = IDCT(blocks[i][j])

    return recon                              #returnerer rekonsturert bilde

image_recon = reconstruct(blocks)

plt.figure()
plt.title("steg 4 test")
plt.imshow(image_recon, cmap='gray')
plt.show()

def verify(reconstructed, original):                    #verifiserer verdiene med original
    N, M = img.shape
    for i in range(N):
        for j in range(M):
            if reconstructed[i,j] == original[i,j]:
                pass
            else:
                print('not equal to original')
                
#verify(image_recon, img)

#steg 5: punktvis divider
Q = [
    [6, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ]
q = [0.1, 0.5, 2, 8, 32]                    #tall parameter q fra steg 7


def divide(blocks, q, Q): #punktvis dividere etter DCT(steg3)
    N, M = img.shape
    Nb, Mb = int(N/8) , int(M/8)
    Qq =  np.multiply(Q,q)
    block_div = np.zeros((Nb,Mb), dtype=np.ndarray)
    
    for i in range(0,Nb):
            for j in range(0,Mb):
                block_div[i][j] = np.round((np.divide(DCT(blocks[i][j]), Qq)))  #bruker steg 3
    
    return block_div



#steg 6: beregn entropi
def connect(blocks):                     #regner invers DCT og setter blokkene sammen
    N, M = img.shape
    Nb, Mb = int(N/8) , int(M/8)
    connected = np.zeros((N,M))

    for i in range (Nb):
        for j in range (Mb):
            x, y = i*8, j*8 
            connected[x:x+8,  y:y+8] = blocks[i][j]

    return connected

d1 = divide(blocks, q[0], Q)
d2 = divide(blocks, q[1], Q)
d3 = divide(blocks, q[2], Q)
d4 = divide(blocks, q[3], Q)
d5 = divide(blocks, q[4], Q)  

def entropi(blocks): #(endret her)
    temporary = np.histogram(np.ravel(blocks), bins = 256)[0]/img.size
    marginal_dist = list(filter(lambda p_val: p_val > 0, np.ravel(temporary)))
    entropi = 0 #-np.sum(np.multiply(marginal_dist, log2(marginal_dist)))
    for i in range (len(marginal_dist)):
        if marginal_dist[i] != 0:
            entropi += marginal_dist[i] * np.log(marginal_dist[i])
    return -entropi

c1 = connect(d1)
c2 = connect(d2)
c3 = connect(d3)
c4 = connect(d4)
c5 = connect(d5)


print('kompresjonsrate=', q[0], 'Entropi=', entropi(c1))
print('kompresjonsrate=', q[1], 'Entropi=', entropi(c2))
print('kompresjonsrate=', q[2], 'Entropi=', entropi(c3))
print('kompresjonsrate=', q[3], 'Entropi=', entropi(c4))
print('kompresjonsrate=', q[4], 'Entropi=', entropi(c5))


#steg 7: (funket ikke, har prøvd å få noe ut)
def reverse_construct(d, q, Q):
    N, M = img.shape
    Nb, Mb = int(N/8) , int(M/8)
    Qq =  np.multiply(Q,q)
    block_new = np.zeros((Nb,Mb), dtype=np.ndarray)
    
    for i in range(0,Nb):
            for j in range(0,Mb):
                block_new[i][j] = reconstruct(np.multiply(blocks[i][j], Qq)) #invers DCT i funksjon reconstruct
    return block_new

img1 = reverse_construct(d1, q[0], Q)
#img2 = reverse_construct(d2, q[1], Q)
#img3 = reverse_construct(d3, q[2], Q)
#img4 = reverse_construct(d4, q[3], Q)
#img5 = reverse_construct(d5, q[4], Q)
#plt.imsave('uio_q=0.1.png', img1, cmap='gray')

"""
#prøvde å få mellon-rsultat bilde, funket ikke
plt.figure()
plt.title("c1, q = 0.1 ")
plt.imshow(c1, cmap='gray')
plt.show()

plt.figure()
plt.title("d1, q = 0.1 ")
plt.imshow(img1, cmap='gray')
plt.show()
"""