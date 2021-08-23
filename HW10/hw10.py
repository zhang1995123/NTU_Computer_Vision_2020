import cv2
import numpy as np

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

def conv(img, kernel):
    x_img, y_img = img.shape
    x_k, y_k = kernel.shape
    res = 0
    for i in range(x_img):
        for j in range(y_img):
            if x_img - i - 1 >= 0 and x_img - i - 1 < x_k and \
               y_img - j - 1 >= 0 and y_img - j - 1 < y_k:
                res += img[i, j] * kernel[x_img - i - 1, y_img - j - 1]
    return res
            
def laplace_1(img_in, threshold):
    k = np.array([[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]])
    x0, y0 = img_in.shape
    xk, yk = k.shape
    res = np.zeros((x0 - xk + 1, y0 - yk + 1))
    x, y = res.shape
    
    for i in range(x):
        for j in range(y):
            tmp = conv(img_in[i:i + xk, j:j + yk], k)
            if tmp >= threshold:
                res[i, j] = 1
            elif tmp <= -threshold:
                res[i, j] = -1
            else:
                res[i, j] = 0
                
    return res 

def laplace_2(img_in, threshold):
    k = np.array([[1, 1, 1],
                 [1, -8, 1],
                 [1, 1, 1]]) / 3
    x0, y0 = img_in.shape
    xk, yk = k.shape
    res = np.zeros((x0 - xk + 1, y0 - yk + 1))
    x, y = res.shape
    
    for i in range(x):
        for j in range(y):
            tmp = conv(img_in[i:i + xk, j:j + yk], k)
            if tmp >= threshold:
                res[i, j] = 1
            elif tmp <= -threshold:
                res[i, j] = -1
            else:
                res[i, j] = 0
    
    return res 

def min_var_laplace(img_in, threshold):
    k = np.array([[2, -1, 2],
                 [-1, -4, -1],
                 [2, -1, 2]]) / 3
    x0, y0 = img_in.shape
    xk, yk = k.shape
    res = np.zeros((x0 - xk + 1, y0 - yk + 1))
    x, y = res.shape
    
    for i in range(x):
        for j in range(y):
            tmp = conv(img_in[i:i + xk, j:j + yk], k)
            if tmp >= threshold:
                res[i, j] = 1
            elif tmp <= -threshold:
                res[i, j] = -1
            else:
                res[i, j] = 0
    
    return res 

def laplace_of_gaussian(img_in, threshold):
    k = np.array([[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
                 [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                 [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                 [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                 [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                 [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
                 [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                 [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                 [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                 [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                 [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])
    x0, y0 = img_in.shape
    xk, yk = k.shape
    res = np.zeros((x0 - xk + 1, y0 - yk + 1))
    x, y = res.shape
    
    for i in range(x):
        for j in range(y):
            tmp = conv(img_in[i:i + xk, j:j + yk], k)
            if tmp >= threshold:
                res[i, j] = 1
            elif tmp <= -threshold:
                res[i, j] = -1
            else:
                res[i, j] = 0
    
    return res

def dif_of_gaussian(img_in, threshold):
    k = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
                 [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                 [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                 [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                 [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                 [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
                 [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                 [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                 [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                 [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                 [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],])
    x0, y0 = img_in.shape
    xk, yk = k.shape
    res = np.zeros((x0 - xk + 1, y0 - yk + 1))
    x, y = res.shape
    
    for i in range(x):
        for j in range(y):
            tmp = conv(img_in[i:i + xk, j:j + yk], k)
            if tmp >= threshold:
                res[i, j] = 1
            elif tmp <= -threshold:
                res[i, j] = -1
            else:
                res[i, j] = 0
    
    return res

# k for edge detection mask dimension
def zero_crossing_detector(img_in, xk, yk):
    x0, y0 = img_in.shape
    res = np.full(img_in.shape, 255, dtype=int) 
    
    for i in range(x0):
        for j in range(y0):
            edge = 255
            if img_in[i, j] == 1:
                for ki in range(-xk // 2 + 1, xk // 2 + 1):
                    for kj in range(-yk // 2 + 1, yk // 2 + 1):
                        if  i + ki >= 0 and i + ki < x0 \
                        and j + kj >= 0 and j + kj < y0:
                            if img_in[i + ki, j + kj] == -1:
                                edge = 0
            res[i, j] = edge
    return res

    
if __name__ == '__main__':
    img_laplace_1 = zero_crossing_detector(laplace_1(img, 15), 3, 3)
    cv2.imwrite('laplace_1.png', img_laplace_1)
    print('Laplace Mask1')
    
    img_laplace_2 = zero_crossing_detector(laplace_2(img, 15), 3, 3)
    cv2.imwrite('laplace_2.png', img_laplace_2)
    print('Laplace Mask2')
    
    img_min_var_laplace = zero_crossing_detector(min_var_laplace(img, 20), 3, 3)
    cv2.imwrite('min_var_laplace.png', img_min_var_laplace)
    print('Minimum variance Laplacian')
    
    img_laplace_of_gaussian = zero_crossing_detector(laplace_of_gaussian(img, 3000), 3, 3)
    cv2.imwrite('laplace_of_gaussian.png', img_laplace_of_gaussian)
    print('Laplace of Gaussian')
    
    img_dif_of_gaussian = zero_crossing_detector(dif_of_gaussian(img, 1), 3, 3)
    cv2.imwrite('dif_of_gaussian.png', img_dif_of_gaussian)
    print('Difference of Gaussian')
