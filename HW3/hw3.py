import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_his(img, x0, y0, save_name):
    his = [0 for i in range(256)]
    for x in range(x0):
        for y in range(y0):
            his[img[x,y]] += 1
    
    # plt.bar(range(0,256), his)
    # plt.savefig(save_name + '.png')
    # plt.show()
    return his

def his_eq(img, his, x0, y0):
    cdf = np.cumsum(his)
    res = np.zeros(img.shape, dtype='uint8')

    for x in range(x0):
        for y in range(y0):
            res[x,y] = round(255.0 * (cdf[img[x,y]]-cdf[0]) / (cdf[-1]-cdf[0]))
    return res

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
x0, y0 = img.shape

# (a) original image and its histogram
img1 = img
# cv2.imwrite('1.png', img1)

# img1_h = plot_his(img1, x0, y0, 'img1_h')

# (b) image with intensity divided by 3 and its histogram
img2 = img // 3
# cv2.imwrite('2.png', img2)

img2_h = plot_his(img2, x0, y0, 'img2_h')

# (c) image after applying histogram equalization to (b) and its histogram
img3 = his_eq(img2, img2_h, x0, y0)
# cv2.imwrite('3.png', img3)

img3_h = plot_his(img3, x0, y0, 'img3_h')
# plt.bar(range(0,256), img3_h)
# plt.savefig('img3_h.png')
