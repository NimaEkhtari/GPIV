import pdal
import numpy as np


# need to blow this up and user a filters.python applied to the Y values

# read in points with pdal
pipeline = [
    {
        "type":"readers.las",
        "filename":"C:/dev/data/PIVDemoFiles/LidarPointClouds/CanadaGlacier_NASA.las"
    }
]
p = pdal.Pipeline(json.dumps(pipeline))
p.validate()
p.execute()
pts = p.arrays[0]
X = pts['X']
Y = pts['Y']
Z = pts['Z']

# multiply each point with 2D gaussian
u = np.array([[0],[0]]) # origin
E = np.array([[1,0],[0,2]]) # covariance
invE = np.linalg.inv(E)
for i in range(X.shape[0]):
    ptVec = np.array([[X[i]],[Y[i]]])
    Y[i] += Yshift * exp(-(1/2) * np.matmul(np.transpose(ptVec-u), np.matmul(invE, (ptVec-u))))

# save points using pdal
