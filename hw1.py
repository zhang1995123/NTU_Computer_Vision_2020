import numpy as np
import cv2

img = cv2.imread('lena.bmp')
x, y, z = img.shape
# print(img.shape) # 512,512,3

# (a) upside-down lena.bmp
# img1 = np.flip(img, axis=0)
img1 = np.zeros((512,512,3))
for m in range(x):
    for n in range(y):
        img1[511-m, n] = img[m,n]
img1 = np.array(img1, dtype=np.uint8)
# cv2.imwrite('1.png', img1)

# (b) right-side-left lena.bmp
# img2 = np.flip(img, axis=1)
img2 = np.zeros((512,512,3))
for m in range(x):
    for n in range(y):
        img2[m, 511-n] = img[m,n]
img2 = np.array(img2, dtype=np.uint8)
# cv2.imwrite('2.png', img2)

# (c) diagonally flip lena.bmp
img3 = img.transpose((1,0,2))
# cv2.imwrite('3.png', img3)

# (f) binarize lena.bmp at 128 to get a binary image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img6 = np.zeros((512,512))
for m in range(x):
    for n in range(y):
        if img[m,n] >= 128:
            img6[m,n] = 255
        else:
            img6[m,n] = 0
# cv2.imwrite('6.png', img6)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
