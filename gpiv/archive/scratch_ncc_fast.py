from piv_option import get_image_arrays
import numpy as np
import matplotlib.pyplot as plt
import json
import time

tptSize = 32
rowStart = 400
colStart = 400

# image and template
fromHeight, fromError, toHeight, toError, transform = get_image_arrays()
tpt = fromHeight[rowStart+1:rowStart+1+tptSize,colStart+1:colStart+1+tptSize]
img = toHeight[rowStart:rowStart+tptSize+2,colStart:colStart+tptSize+2]
# tpt = fromHeight[int(rowStart+tptSize*0.5):int(rowStart+tptSize*1.5),int(colStart+tptSize*0.5):int(colStart+tptSize*1.5)]
# img = toHeight[rowStart:rowStart+tptSize*2,colStart:colStart+tptSize*2]

################
# current faster
################
t0 = time.time()
n = tpt.size
uRange = img.shape[1]-tptSize+1 # columns
vRange = img.shape[0]-tptSize+1 # rows
for i in range(10000):
    nccFast = np.zeros((uRange,vRange))
    tptNormalized = (tpt - np.mean(tpt)) / (np.std(tpt))
    for u in range(uRange): # column
        for v in range(vRange): # row
            imgSub = img[v:v+tptSize,u:u+tptSize]
            imgSubNormalized = (imgSub - np.mean(imgSub)) / (np.std(imgSub) * n)
            nccFast[v,u] = np.sum(tptNormalized * imgSubNormalized)

t1 = time.time()
print(t1-t0)



# jsonNp = nccFast.tolist()
# json.dump(jsonNp, open("nccFST.json", "w"))

# f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,2]})
# axs[0].imshow(nccFast, cmap='gray')
# axs[0].set_title('NCC Fast')
# axs[1].imshow(tpt, cmap='gray')
# axs[1].set_title('Template')
# axs[2].imshow(img, cmap='gray')
# axs[2].set_title('Image')
# plt.show()

# #################
# # running sums and squares NCC
# #################

# # running sum and square
# s = np.zeros((img.shape[0]+1, img.shape[1]+1))
# s2 = s.copy()
# print(s.shape)
# for u in range(1,img.shape[1]+1): # columns
#     for v in range(1,img.shape[0]+1): # rows
#         s[u,v] = img[u-1,v-1] + s[u-1,v] + s[u,v-1] - s[u-1,v-1]
#         s2[u,v] = img[u-1,v-1]**2 + s2[u-1,v] + s2[u,v-1] - s2[u-1,v-1]

# # image energy and sum under the template
# imgE = np.zeros((img.shape[0], img.shape[1]))
# imgS = imgE.copy()
# N = tptSize
# for u in range(1,img.shape[1]-tptSize+1):
#     for v in range(1,img.shape[0]-tptSize+1):
#         imgE[u-1,v-1] = s2[u+N-1,v+N-1] - s2[u-1,v+N-1] - s2[u+N-1,v-1] + s2[u-1,v-1]
#         imgS[u-1,v-1] = s[u+N-1,v+N-1] - s[u-1,v+N-1] - s[u+N-1,v-1] + s[u-1,v-1]

# # ncc
# tptMR = tpt - np.mean(tpt)
# tptD = np.sum(tptMR*tptMR)
# nccFast = np.zeros((img.shape[1]-tptSize+1,img.shape[1]-tptSize+1))
# for u in range(img.shape[1]-tptSize+1):
#     for v in range(img.shape[0]-tptSize+1):
#         imgSub = img[v:v+tptSize,u:u+tptSize]
#         numerator = np.sum(imgSub*tptMR)
#         fbar = imgS[u,v]/(N*N)
#         denominator = np.sqrt((imgE[u,v] - 2*fbar*imgS[u,v] + fbar**2)*tptD)
#         nccFast[u,v] = numerator / denominator

# f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,2]})
# axs[0].imshow(nccFast, cmap='gray')
# axs[0].set_title('NCC Fast')
# axs[1].imshow(tpt, cmap='gray')
# axs[1].set_title('Template')
# axs[2].imshow(img, cmap='gray')
# axs[2].set_title('Image')
# plt.show()