import pdal
import json
import math
import rasterio
import show_option


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


def create_rasters(lasFile, rasterSize, f, t):

    rasterRadius = float(rasterSize)*math.sqrt(0.5)
    if f:
        fromOrTo = 'From'
        rasterFileName = 'from.tif'
        heightFileName = 'fromHeight.tif'
        errorFileName = 'fromError.tif'
    if t:
        fromOrTo = 'To'
        rasterFileName = 'to.tif'
        heightFileName = 'toHeight.tif'
        errorFileName = 'toError.tif'

    print("Generating the '{}' raster".format(fromOrTo))

    # determine the raster bounds that will force the raster to use horizontal coordinates that are clean multiples of the raster resolution
    jsonDict = {
        "pipeline":[
            lasFile,
            {
                "type":"filters.stats",
                "dimensions":"X,Y"
            }
        ]
    }
    jsonString = json.dumps(jsonDict) # converts from a dict to a string for pdal
    pipeline = pdal.Pipeline(jsonString)
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

    # raster points with pdal
    jsonDict = {
        "pipeline": [
            lasFile,
            {
                "resolution": rasterSize,
                "radius": rasterRadius,
                "bounds": bounds,
                "filename": rasterFileName
            }
        ]
    }
    jsonString = json.dumps(jsonDict)
    pipeline = pdal.Pipeline(jsonString)
    pipeline.validate()
    pipeline.execute()

    # save the height and error images as separate tif files
    with rasterio.open(rasterFileName) as src:
        heightRaster =  src.read(3) # read height band to numpy array
        errorRaster = src.read(6)
        profile = src.profile
    
    profile.update(count=1)    
    with rasterio.open(heightFileName, 'w', **profile) as dst:
        dst.write(heightRaster, 1)
    with rasterio.open(errorFileName, 'w', **profile) as dst:
        dst.write(errorRaster, 1)

    # display the height and error rasters
    show_option.show(f, t, True, False, False, [], False, [])
    show_option.show(f, t, False, True, False, [], False, [])
