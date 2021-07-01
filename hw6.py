import numpy as np
import cv2

def downsampled(img):
    out = img.copy()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            out[x,y] = img[x*8,y*8]

    return out

def getValue(img, x, y):
    if x >= img.shape[0] or x < 0 or\
       y >= img.shape[1] or y < 0:
        return 0
    return img[x, y]


def getNeighbors(img, x, y ):
    return [getValue(img, x, y), 
            getValue(img, x+1, y), getValue(img, x, y-1), getValue(img, x-1, y), getValue(img, x, y+1),
            getValue(img, x+1, y+1), getValue(img, x+1, y-1), getValue(img, x-1, y-1), getValue(img, x-1, y+1)]

def hFunc(b, c, d, e):
    if b == c and (d!=b or e!=b):
        return 'q'
    elif b == c and (d==b or e==b):
        return 'r'
    elif b != c:
        return 's'

def fFunc(a1,a2,a3,a4):
    neighbors = [a1, a2, a3, a4]
    if(neighbors.count('r') == len(neighbors)): # all neighbors are equal to r
        return 5
    else:
        return neighbors.count('q')

def Yokoi(neighbors):
    return fFunc(hFunc(neighbors[0],neighbors[1],neighbors[6],neighbors[2]),
                 hFunc(neighbors[0],neighbors[2],neighbors[7],neighbors[3]),
                 hFunc(neighbors[0],neighbors[3],neighbors[8],neighbors[4]),
                 hFunc(neighbors[0],neighbors[4],neighbors[5],neighbors[1]))


img_0 = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
# img_0 = cv2.resize(img_0, (64,64), interpolation=cv2.INTER_NEAREST)
# img_0 = cv2.resize(img_0, (64,64), interpolation=cv2.INTER_LINEAR)
img_0 = downsampled(img_0)
img = img_0.copy()
for i in range(64):
    for j in range(64):
        if img_0[i, j] >= 128:
            img[i, j] = 255
        else:
            img[i, j] = 0

result = [[' ' for x in range(64)] for y in range(64)]

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x, y] != 0:
            result[x][y] = Yokoi(getNeighbors(img, x, y))

result = np.array(result)
# cv2.imwrite('result.png', img)

file = open("Yokoi.txt", "w")
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        file.write(str(result[x, y]))
    file.write('\n')
