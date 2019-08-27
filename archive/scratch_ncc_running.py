from piv_option import get_image_arrays
import numpy as np
import matplotlib.pyplot as plt
import json
import time

tptSize = 128
rowStart = 400
colStart = 400

# image and template
fromHeight, fromError, toHeight, toError, transform = get_image_arrays()
# tpt = fromHeight[rowStart+1:rowStart+1+tptSize,colStart+1:colStart+1+tptSize]
# img = toHeight[rowStart:rowStart+tptSize+2,colStart:colStart+tptSize+2]
tpt = fromHeight[int(rowStart+tptSize*0.5):int(rowStart+tptSize*1.5),int(colStart+tptSize*0.5):int(colStart+tptSize*1.5)]
img = toHeight[rowStart:rowStart+tptSize*2,colStart:colStart+tptSize*2]

#################
# running sums and squares NCC
#################

# Running sum and square. This method is an order of magnitude slower than the sk.image "match_template" function that
# uses FFT in the numerator. However, it is numerically equivalent to brute force NCC, whereas FFT based methods are
# not, and about twice as fast as brute force NCC. This improvement in speed over brute force NCC does not hold when the
# image search area is only a small amount larger than the template. Hence, the brute force method is used for the error
# propagation where the image search area is only two pixels larger in each dimension than the template (note that brute
# force NCC is also faster than FFT based NCC in this case as well).
t0 = time.time()
for i in range(100):
    s = np.zeros((img.shape[0]+1, img.shape[1]+1))
    s2 = s.copy()
    for u in range(1,img.shape[1]+1): # columns
        for v in range(1,img.shape[0]+1): # rows
            s[u,v] = img[u-1,v-1] + s[u-1,v] + s[u,v-1] - s[u-1,v-1]
            s2[u,v] = img[u-1,v-1]**2 + s2[u-1,v] + s2[u,v-1] - s2[u-1,v-1]

    tptZeroMean = tpt  - np.mean(tpt)
    tptZeroMeanSquareSum = np.sum(tptZeroMean**2)
    N = tpt.shape[0]
    M = img.shape[0]

    ncc = np.zeros((M-N+1,M-N+1))
    for u in range(M-N+1):
        for v in range (M-N+1):
            imgSum = s[u+N,v+N] - s[u,v+N] - s[u+N,v] + s[u,v]
            numerator = np.sum(img[u:u+N,v:v+N] * tptZeroMean)
            denominator = np.sqrt(((s2[u+N,v+N] - s2[u,v+N] - s2[u+N,v] + s2[u,v]) - imgSum**2 / (N*N)) * tptZeroMeanSquareSum)
            ncc[u,v] = numerator / denominator
        
t1 = time.time()
print(t1-t0)

# f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,2]})
# axs[0].imshow(ncc, cmap='gray')
# axs[0].set_title('NCC Running Sum')
# axs[1].imshow(tpt, cmap='gray')
# axs[1].set_title('Template')
# axs[2].imshow(img, cmap='gray')
# axs[2].set_title('Image')
# plt.show()

