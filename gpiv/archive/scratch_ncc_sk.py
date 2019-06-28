from piv_option import get_image_arrays
import numpy as np
import matplotlib.pyplot as plt
import json
from skimage.feature import match_template
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


################
# skimage ncc
################
t0 = time.time()
for i in range(100):
    nccSK = match_template(img, tpt)

t1 = time.time()
print(t1-t0)

# jsonNp = nccSK.tolist()
# json.dump(jsonNp, open("nccSK.json", "w"))

# f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,2]})
# axs[0].imshow(nccSK, cmap='gray')
# axs[0].set_title('NCC SK')
# axs[1].imshow(tpt, cmap='gray')
# axs[1].set_title('Template')
# axs[2].imshow(img, cmap='gray')
# axs[2].set_title('Image')
# plt.show()