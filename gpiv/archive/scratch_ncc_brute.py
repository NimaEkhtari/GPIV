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
# brute force NCC
#################
t0 = time.time()
uRange = img.shape[1]-tptSize+1 # columns
vRange = img.shape[0]-tptSize+1 # rows

for i in range(100):
    nccBrute = np.zeros((uRange,vRange))
    meanTpt = np.mean(tpt)
    tptMeanRemoved = tpt-meanTpt
    stdTpt = np.std(tpt)
    n = tpt.size
    for u in range(uRange): # column
        for v in range(vRange): # row
            imgSub = img[v:v+tptSize,u:u+tptSize]
            nccBrute[v,u] = (1/n) * (1/(stdTpt * np.std(imgSub))) * np.sum((imgSub-np.mean(imgSub)) * tptMeanRemoved)

t1 = time.time()
print(t1-t0)

# # save to json
# jsonNp = nccBrute.tolist()
# json.dump(jsonNp, open("nccBrute.json", "w"))

# f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,2]})
# axs[0].imshow(nccBrute, cmap='gray')
# axs[0].set_title('NCC Brute')
# axs[1].imshow(tpt, cmap='gray')
# axs[1].set_title('Template')
# axs[2].imshow(img, cmap='gray')
# axs[2].set_title('Image')
# plt.show()

