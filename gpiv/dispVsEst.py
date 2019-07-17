import json
import numpy as np

with open('piv_origins_offsets.json') as jsonFile:
    disp = json.load(jsonFile)

disp = np.asarray(disp)
xDisp = disp[:,2]
yDisp = disp[:,3]

with open('piv_covariance_matrices.json') as jsonFile:
    cov = json.load(jsonFile)

cov = np.asarray(cov)
xStd = []
yStd = []
for i in range(cov.shape[0]):
    xStd.append(np.sqrt(cov[i][0][0]))
    yStd.append(np.sqrt(cov[i][1][1]))

cmp = np.column_stack((xDisp,yDisp,xStd,yStd))
np.set_printoptions(precision=3, suppress=True)
print(cmp)

