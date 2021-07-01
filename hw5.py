import numpy as np
import cv2

def findMax(input_, x, y, kernel):
    n = kernel.shape[0] // 2
    if x >= n:
        top = x
    else:
        top = n

    if y >= n:
        left = y
    else:
        left = n

    return input_[top-n:top+n+1, left-n:left+n+1].max()

def findMin(input_, x, y, kernel):
    n = kernel.shape[0] // 2
    if x >= n:
        top = x
    else:
        top = n

    if y >= n:
        left = y
    else:
        left = n

    return input_[top-n:top+n+1, left-n:left+n+1].min()

def dilation(img, kernel):
    x0, y0 = img.shape
    n = kernel.shape[0] // 2
    img_d = np.zeros(img.shape, np.uint8)

    for x in range(x0):
        for y in range(y0):
            v = findMax(img, x, y, kernel)
            img_d[x, y] = v                
    return img_d

def erosion(img, kernel):
    x0, y0 = img.shape
    n = kernel.shape[0] // 2
    img_e = np.zeros(img.shape, np.uint8)

    for x in range(x0):
        for y in range(y0):
            v = findMin(img, x, y, kernel)
            img_e[x, y] = v 
    return img_e

def opening(img, kernel):
    img = erosion(img, kernel)
    img = dilation(img, kernel)
    return img

def closing(img, kernel):
    img = dilation(img, kernel)
    img = erosion(img, kernel)
    return img


img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

# cv2.imwrite('binary.png', img)

# kernel is a 3-5-5-5-3 octagon
kernel = np.array([[0,1,1,1,0],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [0,1,1,1,0]], dtype=np.uint8)

# (a) Dilation
img1 = dilation(img, kernel)
cv2.imwrite('dilation.png', img1)

# (b) Erosion
img2 = erosion(img, kernel)
cv2.imwrite('erosion.png', img2)

# (c) Opening
img3 = opening(img, kernel)
cv2.imwrite('open.png', img3)

# (d) Closing
img4 = closing(img, kernel)
cv2.imwrite('close.png', img4)
