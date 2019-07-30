import pdal
import math
import json
import os


inLas = "D:/dev/data/PIVDemoFiles/LidarPointClouds/CanadaGlacier_NASA.las"
rasterSize = 5

rasterRadius = float(rasterSize)*math.sqrt(0.5)

def clean_multiple(num, multiple, direction):
    remainder = abs(num) % multiple
    if remainder == 0:
        return num    
    if direction == 'up':
        if num < 0:
            return num + remainder
        else:
            return num + multiple - remainder
    else:
        if num < 0:
            return num - multiple + remainder
        else:
            return num - remainder


for i in range(0,14,2):
    tMatrix = "1 0 0 {} 0 1 0 {} 0 0 1 0 0 0 0 1".format(i,i)
    outLas = inLas[:-4] + "_Tx{}Ty{}.las".format(i,i)
    outTif = outLas[:-4] + ".tif"

    # shift point cloud in x and y
    jsonDict = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": inLas
            },
            {
                "type": "filters.transformation",
                "matrix": tMatrix
            },
            {
                "type": "writers.las",
                "filename": outLas
            }            
        ]
    }
    jsonString = json.dumps(jsonDict)
    pipeline = pdal.Pipeline(jsonString)
    pipeline.validate()
    pipeline.execute()

    # # determine the raster bounds that will force the raster to use horizontal coordinates that are clean multiples of the raster resolution
    # jsonDict = {
    #     "pipeline":[
    #         outLas,
    #         {
    #             "type":"filters.stats",
    #             "dimensions":"X,Y"
    #         }
    #     ]
    # }
    # jsonString = json.dumps(jsonDict) # converts from a dict to a string for pdal
    # pipeline = pdal.Pipeline(jsonString)
    # pipeline.validate()
    # pipeline.execute()

    # meta = pipeline.metadata
    # meta = json.loads(meta) # converts from a string to a dict

    # minx = meta.get('metadata').get('readers.las')[0].get('minx')
    # maxx = meta.get('metadata').get('readers.las')[0].get('maxx')
    # miny = meta.get('metadata').get('readers.las')[0].get('miny')
    # maxy = meta.get('metadata').get('readers.las')[0].get('maxy')

    # minx = clean_multiple(minx, float(rasterSize), 'down')
    # maxx = clean_multiple(maxx, float(rasterSize), 'up')
    # miny = clean_multiple(miny, float(rasterSize), 'down')
    # maxy = clean_multiple(maxy, float(rasterSize), 'up')

    # bounds = "([{},{}],[{},{}])".format(minx,maxx,miny,maxy)

    # raster points with pdal
    jsonDict = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": outLas
            },
            {
                "type": "writers.gdal",
                "gdaldriver": "GTiff",
                "resolution": rasterSize,
                "radius": rasterRadius,
                "bounds": "([21925,24435],[40825,43225])",                
                "window_size": 5,
                "nodata": 0,
                "output_type": "idw",
                "filename": outTif
            }
        ]
    }
    jsonString = json.dumps(jsonDict)
    pipeline = pdal.Pipeline(jsonString)
    pipeline.validate()
    pipeline.execute()



