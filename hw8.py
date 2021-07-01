import numpy as np
import cv2
import math

def dilation(img, kernel):
    row, col = img.shape # original image
    res = np.zeros(img.shape, dtype = 'int32')
    
    for x in range(row):
        for y in range(col):
            max_value = 0
            for k in kernel:
                i, j = k
                if  x+i >= 0 and x+i < row and y+j >= 0 and y+j < col: 
                    max_value = max(max_value, img[x+i, y+j])
            res[x, y] = max_value      
    return res 

def erosion(img, kernel):
    row, col = img.shape # original image
    res = np.zeros(img.shape, dtype = 'int32')
    
    for x in range(row):
        for y in range(col):
            min_value = 255
            for k in kernel:
                i, j = k
                if  x+i >= 0 and x+i < row and y+j >= 0 and y+j < col: 
                    min_value = min(min_value, img[x+i, y+j])
            res[x, y] = min_value
    return res 

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def gaussian_noise(img_in, mu, sigma, amp):
    return img_in + amp * np.random.normal(mu, sigma, img_in.shape)

def salt_pepper_noise(img_in, prob):
    distribution_map = np.random.uniform(0, 1, img.shape)
    res = np.copy(img_in)
    row, col = img_in.shape
    
    for i in range(row):
        for j in range(col):
            if distribution_map[i, j] < prob:
                res[i, j] = 0
            elif distribution_map[i, j] > 1 - prob: 
                res[i, j] = 255
    return res

def box_filter(img_in, box_size):
    kernel = []
    for i in range(-box_size // 2, box_size // 2):
        for j in range(-box_size // 2, box_size // 2):
            kernel.append([i, j])
    
    row, col = img_in.shape
    res = np.zeros(img_in.shape)
    scale = box_size * box_size
    
    for i in range(row):
        for j in range(col):
            val = 0
            for k in kernel:
                ki, kj = k
                if i+ki >= 0 and i+ki < row and \
                    j+kj >= 0 and j+kj < col:
                    val += img_in[i+ki, j+kj]
            res[i, j] = val / scale
    return res

def median_filter(img_in, kernel_size):
    kernel = []
    for i in range(-kernel_size // 2, kernel_size // 2):
        for j in range(-kernel_size // 2, kernel_size // 2):
            kernel.append([i, j])
    
    row, col = img_in.shape
    res = np.zeros(img_in.shape)
    
    for i in range(row):
        for j in range(col):
            vals = []
            for k in kernel:
                ki, kj = k
                if i+ki >= 0 and i+ki < row and \
                   j+kj >= 0 and j+kj < col:
                    vals.append(img_in[i+ki, j+kj])
            res[i, j] = np.median(vals)
    return res


def snr(signal, noise, img_name):
    # signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    # noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    row, col = signal.shape
    n = row * col

    sum_signal = 0.
    mu_s = 0.
    sum_vs = 0.
    v_s = 0.
    sum_noise = 0.
    mu_n = 0.
    sum_vn = 0.
    v_n = 0.
    
    for i in range(row):
        for j in range(col):
            sum_signal += float(signal[i, j])
            sum_noise += float(noise[i, j]) - float(signal[i, j])
    mu_s = sum_signal / n
    mu_n = sum_noise / n

    for i in range(row):
        for j in range(col):
            sum_vs += (float(signal[i, j]) - float(mu_s))**2
            sum_vn += (float(noise[i, j]) - float(signal[i, j]) - float(mu_n))**2
    v_s = sum_vs / n
    v_n = sum_vn / n

    res = 20 * np.log10(np.sqrt(v_s) / np.sqrt(v_n))
    print('SNR of %s is %f' %(img_name, res))



img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
test_n = cv2.imread('median_5x5.bmp', cv2.IMREAD_GRAYSCALE)
print(snr(img, test_n, 'test'))

kernel = [         [-1, -2], [0, -2], [1, -2],
         [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
          [-2, 0],  [-1, 0], [0, 0], [1, 0], [2, 0],
          [-2, 1],  [-1, 1], [0, 1], [1, 1], [2, 1],
                    [-1, 2], [0, 2], [1, 2]
         ]

### (a) gaussian noise
print('------ a ------')
g_10 = gaussian_noise(img, 0, 1, 10)
g_30 = gaussian_noise(img, 0, 1, 30)
snr(img, g_10, 'g_10')
snr(img, g_30, 'g_30')
cv2.imwrite('./a_gaussian_10.png', g_10) 
cv2.imwrite('./a_gaussian_30.png', g_30) 

### (b) salt-and-pepper noise
print('------ b ------')
sp_05 = salt_pepper_noise(img, 0.05)
sp_10 = salt_pepper_noise(img, 0.1)
snr(img, sp_05, 'sp_05')
snr(img, sp_10, 'sp_10')
cv2.imwrite('./b_sp_05.png', sp_05) 
cv2.imwrite('./b_sp_10.png', sp_10)

### (c) box filter
print('------ c ------')
temp = box_filter(g_10, 3)
snr(img, temp, 'box_3_g_10')
cv2.imwrite('./box_3_g_10.png', temp)

temp = box_filter(g_10, 5)
snr(img, temp, 'box_5_g_10')
cv2.imwrite('./box_5_g_10.png', temp)

temp = box_filter(g_30, 3)
snr(img, temp, 'box_3_g_30')
cv2.imwrite('./box_3_g_30.png', temp)

temp = box_filter(g_30, 5)
snr(img, temp, 'box_5_g_30')
cv2.imwrite('./box_5_g_30.png', temp)

temp = box_filter(sp_05, 3)
snr(img, temp, 'box_3_sp_05')
cv2.imwrite('./box_3_sp_05.png', temp)

temp = box_filter(sp_05, 5)
snr(img, temp, 'box_5_sp_05')
cv2.imwrite('./box_5_sp_05.png', temp)

temp = box_filter(sp_10, 3)
snr(img, temp, 'box_3_sp_10')
cv2.imwrite('./box_3_sp_10.png', temp)

temp = box_filter(sp_10, 5)
snr(img, temp, 'box_5_sp_10')
cv2.imwrite('./box_5_sp_10.png', temp)

### (d) median filter
print('------ d ------')
temp = median_filter(g_10, 3)
snr(img, temp, 'median_3_g_10')
cv2.imwrite('./median_3_g_10.png', temp)

temp = median_filter(g_10, 5)
snr(img, temp, 'median_5_g_10')
cv2.imwrite('./median_5_g_10.png', temp)

temp = median_filter(g_30, 3)
snr(img, temp, 'median_3_g_30')
cv2.imwrite('./median_3_g_30.png', temp)

temp = median_filter(g_30, 5)
snr(img, temp, 'median_5_g_30')
cv2.imwrite('./median_5_g_30.png', temp)

temp = median_filter(sp_05, 3)
snr(img, temp, 'median_3_sp_05')
cv2.imwrite('./median_3_sp_05.png', temp)

temp = median_filter(sp_05, 5)
snr(img, temp, 'median_5_sp_05')
cv2.imwrite('./median_5_sp_05.png', temp)

temp = median_filter(sp_10, 3)
snr(img, temp, 'median_3_sp_10')
cv2.imwrite('./median_3_sp_10.png', temp)

temp = median_filter(sp_10, 5)
snr(img, temp, 'median_5_sp_10')
cv2.imwrite('./median_5_sp_10.png', temp)

### (e) open, close
print('------ e ------')

temp = closing(opening(g_10, kernel), kernel)
snr(img, temp, 'opening-then-closing_g_10')
cv2.imwrite('./opening-then-closing_g_10.png', temp)

temp = closing(opening(g_30, kernel), kernel)
snr(img, temp, 'opening-then-closing_g_30')
cv2.imwrite('./opening-then-closing_g_30.png', temp)

temp = closing(opening(sp_05, kernel), kernel)
snr(img, temp, 'opening-then-closing_sp_05')
cv2.imwrite('./opening-then-closing_sp_05.png', temp)

temp = closing(opening(sp_10, kernel), kernel)
snr(img, temp, 'opening-then-closing_sp_10')
cv2.imwrite('./opening-then-closing_sp_10.png', temp)

temp = opening(closing(g_10, kernel), kernel)
snr(img, temp, 'closing-then-opening_g_10')
cv2.imwrite('./closing-then-opening_g_10.png', temp)

temp = opening(closing(g_30, kernel), kernel)
snr(img, temp, 'closing-then-opening_g_30')
cv2.imwrite('./closing-then-opening_g_30.png', temp)

temp = opening(closing(sp_05, kernel), kernel)
snr(img, temp, 'closing-then-opening_sp_05')
cv2.imwrite('./closing-then-opening_sp_05.png', temp)

temp = opening(closing(sp_10, kernel), kernel)
snr(img, temp, 'closing-then-opening_sp_10')
cv2.imwrite('./closing-then-opening_sp_10.png', temp)
