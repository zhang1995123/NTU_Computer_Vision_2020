import numpy as np
import cv2
import matplotlib.pyplot as plt

def dilation(img, kernel):
    x0, y0 = img.shape
    n = kernel.shape[0] // 2
    img_d = np.zeros(img.shape, np.uint8)

    for x in range(n, x0-n):
        for y in range(n, y0-n):
            if img[x, y] > 0:
                img_d[x-n:x+n+1, y-n:y+n+1] += kernel

    for x in range(x0):
        for y in range(y0):
            if img_d[x, y] > 0:
                img_d[x, y] = 255
            else:
                img_d[x, y] = 0                   
    return img_d

def erosion(img, kernel):
    x0, y0 = img.shape
    n = kernel.shape[0] // 2
    total_num = sum(sum(kernel))
    img_e = np.zeros(img.shape, np.uint8)

    for x in range(n, x0-n):
        for y in range(n, y0-n):
            focus = img[x-n:x+n+1, y-n:y+n+1] / 255
            result = focus * kernel
            num = sum(sum(result))
            if num == total_num:
                img_e[x,y] = 255
    return img_e

def opening(img, kernel):
    img = erosion(img, kernel)
    img = dilation(img, kernel)
    return img

def closing(img, kernel):
    img = dilation(img, kernel)
    img = erosion(img, kernel)
    return img

def hit_and_miss(img, kj, kk):
    img_hm = np.zeros(img.shape, np.uint8)
    img_com = 255 - img
    img_ero1 = erosion(img, kernel=kj)
    img_ero2 = erosion(img_com, kernel=kk)
    # img_res = (((img_ero1 + img_ero2)) / 2 == 255) * 255
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img_ero1[x,y] == 255 and img_ero2[x,y] == 255:
                img_hm[x,y] = 255
                print('-')
            else:
                img_hm[x,y] = 0
    
    return img_hm


img_o = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
x0, y0 = img_o.shape
# binary image
img = np.zeros((x0, y0))
for x in range(x0):
    for y in range(y0):
        if img_o[x,y] >= 128:
            img[x,y] = 255
        else:
            img[x,y] = 0

# cv2.imwrite('binary.png', img)

# kernel is a 3-5-5-5-3 octagon
kernel = np.array([[0,1,1,1,0],
				   [1,1,1,1,1],
				   [1,1,1,1,1],
				   [1,1,1,1,1],
				   [0,1,1,1,0]], dtype=np.uint8)
# k = np.ones((5,5),np.uint8)

# (a) Dilation
img1 = dilation(img, kernel)
cv2.imwrite('dilation.png', img1)

# img_ = cv2.dilate(img, k, iterations=1)
# cv2.imwrite('dilation.png', img_)

# (b) Erosion
img2 = erosion(img, kernel)
cv2.imwrite('erosion.png', img2)

# img_ = cv2.erode(img, k, iterations=1)
# cv2.imwrite('erosion.png', img_)

# (c) Opening
img3 = opening(img, kernel)
cv2.imwrite('open.png', img3)

# (d) Closing
img4 = closing(img, kernel)
cv2.imwrite('close.png', img4)

# (e) Hit-and-miss transform
kernel_j = np.array([[0,0,0,0,0],
					 [0,0,0,0,0],
					 [1,1,0,0,0],
					 [0,1,0,0,0],
					 [0,0,0,0,0]])

kernel_k = np.array([[0,0,0,0,0],
					 [0,1,1,0,0],
					 [0,0,1,0,0],
					 [0,0,0,0,0],
					 [0,0,0,0,0]])

img5 = hit_and_miss(img, kernel_j, kernel_k)
cv2.imwrite('h&m.png', img5)
