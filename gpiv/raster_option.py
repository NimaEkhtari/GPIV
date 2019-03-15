import pdal
import json
import math


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


def create_rasters(fromLAS, toLAS, rasterSize):

    rasterRadius = float(rasterSize)*math.sqrt(0.5)

    # determined the raster bounds that will force the 'from' raster to use horizontal coordinates that are clean multiples of the raster resolution
    fromJson = {
        "pipeline":[
            fromLAS,
            {
                "type":"filters.stats",
                "dimensions":"X,Y"
            }
        ]
    }
    fromJson = json.dumps(fromJson) # converts from a dict to a string for pdal
    pipeline = pdal.Pipeline(fromJson)
    pipeline.validate()
    pipeline.execute()

    meta = pipeline.metadata
    meta = json.loads(meta) # converts from a string to a dict

    minx = meta.get('metadata').get('readers.las')[0].get('minx')
    maxx = meta.get('metadata').get('readers.las')[0].get('maxx')
    miny = meta.get('metadata').get('readers.las')[0].get('miny')
    maxy = meta.get('metadata').get('readers.las')[0].get('maxy')

    minx = clean_multiple(minx, float(rasterSize), 'down')
    maxx = clean_multiple(maxx, float(rasterSize), 'up')
    miny = clean_multiple(miny, float(rasterSize), 'down')
    maxy = clean_multiple(maxy, float(rasterSize), 'up')

    bounds = "([" + str(minx) + "," + str(maxx) + "],[" + str(miny) + "," + str(maxy) + "])"

    # raster 'from' points with pdal
    fromRaster = {
        "pipeline": [
            fromLAS,
            {
                "resolution": rasterSize,
                "radius": rasterRadius,
                "bounds": bounds,
                "filename": "from.tif"
            }
        ]
    }
    fromRaster = json.dumps(fromRaster)
    pipeline = pdal.Pipeline(fromRaster)
    pipeline.validate()
    pipeline.execute()

    # determined the raster bounds that will force the 'to' raster to use horizontal coordinates that are clean multiples of the raster resolution
    toJson = {
        "pipeline":[
            toLAS,
            {
                "type":"filters.stats",
                "dimensions":"X,Y"
            }
        ]
    }
    toJson = json.dumps(toJson) # converts from a dict to a string for pdal
    pipeline = pdal.Pipeline(toJson)
    pipeline.validate()
    pipeline.execute()

    meta = pipeline.metadata
    meta = json.loads(meta) # converts from a string to a dict

    minx = meta.get('metadata').get('readers.las')[0].get('minx')
    maxx = meta.get('metadata').get('readers.las')[0].get('maxx')
    miny = meta.get('metadata').get('readers.las')[0].get('miny')
    maxy = meta.get('metadata').get('readers.las')[0].get('maxy')

    minx = clean_multiple(minx, float(rasterSize), 'down')
    maxx = clean_multiple(maxx, float(rasterSize), 'up')
    miny = clean_multiple(miny, float(rasterSize), 'down')
    maxy = clean_multiple(maxy, float(rasterSize), 'up')

    bounds = "([" + str(minx) + "," + str(maxx) + "],[" + str(miny) + "," + str(maxy) + "])"

    # raster 'to' points with pdal
    toRaster = {
        "pipeline": [
            toLAS,
            {
                "resolution": rasterSize,
                "radius": rasterRadius,
                "bounds": bounds,
                "filename": "to.tif"
            }
        ]
    }
    toRaster = json.dumps(toRaster)
    pipeline = pdal.Pipeline(toRaster)
    pipeline.validate()
    pipeline.execute()
