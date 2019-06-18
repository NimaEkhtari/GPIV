from piv_option import get_image_arrays
import numpy as np
import matplotlib.pyplot as plt
import json

tptSize = 64
rowStart = 400
colStart = 400

# image and template
fromHeight, fromError, toHeight, toError, transform = get_image_arrays()
tpt = fromHeight[int(rowStart+tptSize*0.5):int(rowStart+tptSize*1.5),int(colStart+tptSize*0.5):int(colStart+tptSize*1.5)]
img = toHeight[rowStart:rowStart+tptSize*2,colStart:colStart+tptSize*2]
print(img.shape)
print(tpt.shape)

#################
# brute force NCC
#################
uRange = img.shape[1]-tpt.shape[1]+1 # columns
vRange = img.shape[0]-tpt.shape[0]+1 # rows

nccBrute = np.zeros((uRange,vRange))
meanTpt = np.mean(tpt)
stdTpt = np.std(tpt)
n = tpt.size
# print(meanTpt)
# print(stdTpt)
# print(n)
for v in range(uRange): # row
    for u in range(vRange): # column
        imgSub = img[v:v+tptSize,u:u+tptSize]
        meanImgSub = np.mean(imgSub)
        stdImgSub = np.std(imgSub)
        nccBrute[v,u] = (1/n) * (1/(stdTpt * stdImgSub)) * np.sum((imgSub-meanImgSub) * (tpt-meanTpt))

# save to json
jsonNp = nccBrute.tolist()
json.dump(jsonNp, open("nccBrute.json", "w"))

f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,2]})
axs[0].imshow(nccBrute, cmap='gray')
axs[0].set_title('NCC')
axs[1].imshow(tpt, cmap='gray')
axs[1].set_title('Template')
axs[2].imshow(img, cmap='gray')
axs[2].set_title('Image')
plt.show()


#################
# faster NCC
#################
uRange = img.shape[1]-tpt.shape[1]+1 # columns
vRange = img.shape[0]-tpt.shape[0]+1 # rows

s = np.zeros((uRange,vRange))
s2 = np.zeros((uRange,vRange))

# running sum and square sum
s[0,0] = img[0,0]
s2[0,0] = img[0,0]**2
for v in range(1,img.shape[1]):
    s[0,v] = img[0,v] + s[0,v-1]
    s2[0,v] = img[0,v]**2 + s2[0,v-1]
for u in range(1,img.shape[0]):
    s[u,0] = img[u,0] + s[u-1,0]
    s2[u,0] = img[u,0]**2 + s2[u-1,0]
for v in range(1,img.shape[1]):
    for u in range(1,img.shape[0]):
        s[u,v] = img[u,v] + s[u-1,v] + s[u,v-1] - s[u-1,v-1]
        s2[u,v] = img[u,v]**2 + s2[u-1,v] + s2[u,v-1] - s2[u-1,v-1]

# ncc





