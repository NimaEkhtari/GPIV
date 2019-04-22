import pdal
import json
import math
import rasterio


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

    print("Generating the 'From' raster")

    # determine the raster bounds that will force the 'from' raster to use horizontal coordinates that are clean multiples of the raster resolution
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

    # save the 'from' height and error images as separate tif files
    with rasterio.open('from.tif') as src:
        fromHeight =  src.read(3) # read height band to numpy array
        fromError = src.read(6)
        profile = src.profile
    
    profile.update(count=1)    
    with rasterio.open('fromHeight.tif', 'w', **profile) as dst:
        dst.write(fromHeight, 1)
    with rasterio.open('fromError.tif', 'w', **profile) as dst:
        dst.write(fromError, 1)


    print("Generating the 'To' raster")
    # determine the raster bounds that will force the 'to' raster to use horizontal coordinates that are clean multiples of the raster resolution
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

    # save the 'to' height and error images as separate tif files
    with rasterio.open('to.tif') as src:
        toHeight =  src.read(3) # read height band to numpy array
        toError = src.read(6)    
        profile = src.profile
    
    profile.update(count=1)    
    with rasterio.open('toHeight.tif', 'w', **profile) as dst:
        dst.write(toHeight, 1)
    with rasterio.open('toError.tif', 'w', **profile) as dst:
        dst.write(toError, 1)
