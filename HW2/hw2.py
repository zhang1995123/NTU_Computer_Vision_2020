import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
x0, y0 = img.shape

# (a) a binary image (threshold at 128)
img1 = np.zeros((x0, y0))
for x in range(x0):
    for y in range(y0):
        if img[x,y] >= 128:
            img1[x,y] = 255
        else:
            img1[x,y] = 0
# # cv2.imwrite('1.png', img1)
# cv2.imshow('test', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# (b) a histogram
his = [0 for i in range(256)]

for x in range(x0):
    for y in range(y0):
        his[img[x,y]] += 1

# plt.bar(range(0,256), his)
# # plt.savefig('2.png')
# plt.show()

# (c) connected components(regions with + at centroid, bounding box)
img_pad = np.pad(img1, pad_width=[(1,1),(1,1)], mode='constant', constant_values=0)
temp = np.array(img_pad, dtype=np.uint16)
label = 0

for x in range(x0):
    for y in range(y0):
        if img1[x,y] == 255:
            up = temp[x, y+1]
            left = temp[x+1, y]
            
            if up == left == 0:
                label = label + 1
                temp[x+1, y+1] = label
            elif up == 0 and left != 0:
                temp[x+1, y+1] = left
            elif up != 0 and left == 0:
                temp[x+1, y+1] == up
            elif up == left and up != 0:
                temp[x+1, y+1] = up
            else:
                a = min(up, left)
                b = max(up, left)
                temp[x+1, y+1] = a

for x in range(x0):
    for y in range(y0):
        up = temp[x, y+1]
        left = temp[x+1, y]

        if up != 0 and left != 0:
            a = min(up, left)
            b = max(up, left)
            temp[x+1, y+1] = a

# number filter
counter = [0 for i in range(label+1)]
for x in range(x0):
    for y in range(y0):
        val = temp[x, y]
        counter[val] += 1

label_list = []
for idx, num in enumerate(counter):
    if idx != 0 and num >= 500:
        label_list.append(idx)

# bounding box
box = {}
for label in label_list:
    # up, down, left, right
    box[label] = [600, 0, 600, 0]

for y, row in enumerate(temp):
    for x, val in enumerate(row):
        if val in label_list:
            temp[x,y] = 255
            if y < box[val][0]:
                box[val][0] = y
            if y > box[val][1]:
                box[val][1] = y
            if x < box[val][2]:
                box[val][2] = x
            if x > box[val][3]:
                box[val][3] = x
        else:
            temp[x,y] = 0
print(box)
img3 = img1.copy().astype(np.uint8)
img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2RGB)
for b in box:
    up = box[b][0]
    down = box[b][1]
    left = box[b][2]
    right = box[b][3]
    # box
    cv2.rectangle(img3, (left, up), (right, down), (255,0,0),3)
    # +
    width = 12
    x_mid = (left + right) // 2
    y_mid = (up + down) // 2
    x_left = x_mid - width
    x_right = x_mid + width
    y_up = y_mid + width
    y_down = y_mid - width
    cv2.rectangle(img3,(x_left, y_mid),(x_right, y_mid), (0,0,255), 2)
    cv2.rectangle(img3,(x_mid, y_up),(x_mid, y_down), (0,0,255), 2)


# cv2.imwrite('3.png', img3)
cv2.imshow('test', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
