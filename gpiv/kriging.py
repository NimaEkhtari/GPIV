import pdal
import json
import numpy as np
from pykrige.ok import OrdinaryKriging

# get point cloud in numpy form
pipeline = [
    {
        "type":"readers.las",
        "filename":"C:/dev/data/PIVDemoFiles/LidarPointClouds/CanadaGlacier_NCALM.las"
    }
]
p = pdal.Pipeline(json.dumps(pipeline))
p.validate()
p.execute()
las = np.asarray(p.arrays[0])
tst = np.concatenate(las, axis=0)
print(tst.shape)
# xyz = las[:,0:3]

# OK = OrdinaryKriging(xyz[:, 0], xyz[:, 1], xyz[:, 2], variogram_model='linear', verbose=True, enable_plotting=True)
