import numpy as np
import cv2

def downsample(img, n):
    x, y = img.shape
    row, col = x//8, y//8
    out = np.zeros((row, col), np.int)
    for x in range(row):
        for y in range(col):
            out[x, y] = img[x*n, y*n]
    return out

def rounding(Float):
    base = int(Float)
    rem = Float-base
    return base+1 if rem>=0.5 else base

def padding(pixels, layer, mode='black',**kwargs):
    ori_r, ori_c = pixels.shape
    rol,col = ori_r+2*layer, ori_c+2*layer
    out = np.ndarray((rol,col),int).astype('uint8')
    for r in range(ori_r): # copying part
        for c in range(ori_c):
            out[layer+r,layer+c]=pixels[r,c]
    color = 0 if mode=="black" else 255 if mode=="white" else kwargs['color'] if mode=='single' else None
    assert mode=='border' or color!=None, 'illegal mode "{}"'.format(mode)
    if mode=='border':
        for c in range(ori_c): #up
            color = pixels[0,c]
            for l in range(layer):
                new_c = c+layer
                out[l,new_c] = color
        for c in range(ori_c): #down
            color = pixels[ori_r-1,c]
            for l in range(layer):
                new_c = c+layer
                out[layer+ori_r+l,new_c] = color
        for r in range(ori_r): #left
            color = pixels[r,0]
            for l in range(layer):
                new_r = r+layer
                out[new_r,l] = color
        for r in range(ori_r): #right
            color = pixels[r,ori_c-1]
            for l in range(layer):
                new_r = r+layer
                out[new_r,layer+ori_c+l] = color
        for r in range(layer):
            for c in range(layer):# up-left
                out[r,c] = pixels[0,0]
            for c in range(layer+ori_c, layer+ori_c+layer): #up-right
                out[r,c] = pixels[0,ori_c-1]
        for r in range(layer+ori_r, layer+ori_r+layer):
            for c in range(layer):# down-left
                out[r,c] = pixels[ori_r-1,0]
            for c in range(layer+ori_c, layer+ori_c+layer): #down-right
                out[r,c] = pixels[ori_r-1,ori_c-1]
    else:
        for c in range(ori_c): #up
            for l in range(layer):
                new_c = c+layer
                out[l,new_c] = color
        for c in range(ori_c): #down
            for l in range(layer):
                new_c = c+layer
                out[layer+ori_r+l,new_c] = color
        for r in range(ori_r): #left
            for l in range(layer):
                new_r = r+layer
                out[new_r,l] = color
        for r in range(ori_r): #right
            for l in range(layer):
                new_r = r+layer
                out[new_r,layer+ori_c+l] = color
        for r in range(layer):
            for c in range(layer):# up-left
                out[r,c] = color
            for c in range(layer+ori_c, layer+ori_c+layer): #up-right
                out[r,c] = color
        for r in range(layer+ori_r, layer+ori_r+layer):
            for c in range(layer):# down-left
                out[r,c] = color
            for c in range(layer+ori_c, layer+ori_c+layer): #down-right
                out[r,c] = color
    return out

def count_yokoi(array):
    def yokoi_aux(b,c,d,e):
        if b!=c:
            return 's'
        elif d==b and e==b:
            return 'r'
        else:
            return 'q'
    pad=padding(array, 1, mode='black')
    row,col = array.shape
    out=np.zeros((row,col), int) # no need uint8
    for r in range(row): 
        for c in range(col): 
            rp,cp = r+1, c+1 # in padding coordinate
            if array[r,c]==0:
                out[r,c]=0
                continue
            a1=yokoi_aux(pad[rp,cp], pad[rp,cp+1], pad[rp-1,cp+1], pad[rp-1,cp])
            a2=yokoi_aux(pad[rp,cp], pad[rp-1,cp], pad[rp-1,cp-1], pad[rp,cp-1])
            a3=yokoi_aux(pad[rp,cp], pad[rp,cp-1], pad[rp+1,cp-1], pad[rp+1,cp])
            a4=yokoi_aux(pad[rp,cp], pad[rp+1,cp], pad[rp+1,cp+1], pad[rp,cp+1])
            judge=[a1,a2,a3,a4]
            if judge.count('r')==4:
                out[r,c]=5
            else:
                out[r,c]=judge.count('q')
    return out

def pair_relationship(yokoi, m=1):
    row, col = yokoi.shape
    pad = padding(yokoi, 1, mode='black') #padding to avoid border problem
    out = np.zeros((row,col),int)
    for r in range(row):
        for c in range(col):
            rp,cp = r+1,c+1
            if pad[rp,cp]==m: # itself is 'm'
                neighbor = pad[rp,cp+1], pad[rp-1,cp], pad[rp,cp-1], pad[rp+1,cp]
                if neighbor.count(m)>=1: # at least one neighbor is also 'm'
                    out[r,c] = 1 # i.e. mark as "p"
    return out        

def connected_shrink(image, yokoi, pair): #once, recursive
    assert image.shape==yokoi.shape==pair.shape, 'wrong input format'
    row, col = image.shape
    out=image.copy()
    def cs_aux(b,c,d,e):# i.e. yokoi label 'q'
        return 1 if b==c and (d!=b or e!=b) else 0 
    pad=padding(image, 1, mode='black')
    for r in range(row): 
        for c in range(col): 
            if out[r,c]==0:
                continue
            elif pair[r,c]!=1:
                #not candidate to be shrink
                continue
            # remaining pixels are non-zero and edge
            rp,cp = r+1, c+1
            a1=cs_aux(pad[rp,cp], pad[rp,cp+1], pad[rp-1,cp+1], pad[rp-1,cp])
            a2=cs_aux(pad[rp,cp], pad[rp-1,cp], pad[rp-1,cp-1], pad[rp,cp-1])
            a3=cs_aux(pad[rp,cp], pad[rp,cp-1], pad[rp+1,cp-1], pad[rp+1,cp])
            a4=cs_aux(pad[rp,cp], pad[rp+1,cp], pad[rp+1,cp+1], pad[rp,cp+1])
            judge=[a1,a2,a3,a4]
            if judge.count(1)==1: # if exact 1 side if corner
                out[r,c]=0 # the pixel is gone
                pad[rp,cp]=0 # this line is super important, otherwise the Alg falis
            else:
                pass  ## identical to out[r,c]=255
    return out


img_0 = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
# Downsample
img_0 = downsample(img_0, 8)
img = img_0.copy()
# Binarize 
for i in range(64):
    for j in range(64):
        if img_0[i, j] >= 128:
            img[i, j] = 255
        else:
            img[i, j] = 0

res = img
last = None

for iters in range(10):
    print(iters)
    yokoi = count_yokoi(res)
    pair = pair_relationship(yokoi)
    res = connected_shrink(res, yokoi, pair)
    if np.array_equal(last, res):
        break
    last = res

    cv2.imwrite(str(iters)+'.png', res)
