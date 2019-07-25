import math
import json
import numpy as np
import matplotlib.pyplot as plt


with open('piv_origins_offsets.json') as jsonFile:
    disp = json.load(jsonFile)
disp = np.asarray(disp)
xDisp = disp[:,2]
yDisp = disp[:,3]
mDisp = np.sqrt(xDisp**2 + yDisp**2) # magnitude displacement
print('x std = {}'.format(np.std(xDisp)))
print('y std = {}'.format(np.std(yDisp)))

with open('piv_covariance_matrices.json') as jsonFile:
    cov = json.load(jsonFile)
cov = np.asarray(cov)
xStd = []
yStd = []
semiMajor = []
for i in range(cov.shape[0]):
    xStd.append(np.sqrt(cov[i][0][0]))
    yStd.append(np.sqrt(cov[i][1][1]))
    eigenVals, eigenVecs = np.linalg.eig(cov[i])
    idxMax = np.argmax(eigenVals)
    semiMajor.append(math.sqrt(2.298*eigenVals[idxMax])) # scale factor of 2.298 to create a 68% confidence ellipse

print('Mean est x std = {}'.format(np.mean(xStd)))
print('Mean est y std = {}'.format(np.mean(yStd)))

comp = np.column_stack((xDisp,yDisp,mDisp,xStd,yStd,semiMajor))
np.set_printoptions(precision=3, suppress=True)
print(comp)

xCnt = 0
yCnt = 0
mCnt = 0
num = comp.shape[0]
for i in range(num):
    if (abs(comp[i,0]) <= comp[i,3]):
        xCnt += 1
    if (abs(comp[i,1]) <= comp[i,4]):
        yCnt += 1
    if (abs(comp[i,2]) <= comp[i,5]):
        mCnt += 1
xPct = xCnt / num
yPct = yCnt / num
mPct = mCnt / num
print('X percentage = {}'.format(xPct))
print('Y percentage = {}'.format(yPct))
print('Magnitude percentage = {}'.format(mPct))

# fig = plt.figure()
# ax1 = plt.subplot(1, 2, 1)
# ax2 = plt.subplot(1, 2, 2)
# plt.sca(ax1)
# ax1.set_title('X Displacement')
# plt.hist(xDisp)
# plt.sca(ax2)
# ax2.set_title('X Est Error')
# plt.hist(xStd)
# plt.show()