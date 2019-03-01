import rasterio
import rasterio.plot
import rasterio.mask
from shapely import geometry
import numpy as np

def get_arrays():    
    # read in the from and to images as numpy arrays (currently assumes multiple layers in the from and to image files)
    fromRaster = rasterio.open('from.tif')
    # fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
    # fromStd = fromRaster.read(6, masked=True)
    toRaster = rasterio.open('to.tif')
    # toHeight = toRaster.read(3, masked=True)
    # toStd = toRaster.read(6, masked=True) 

    # create a polygon defining the extents of the geospatial overlap
    fromLRBT = list(rasterio.plot.plotting_extent(fromRaster)) # LRBT = [left, right, bottom, top]
    toLRBT = list(rasterio.plot.plotting_extent(toRaster))
    extentsLRBT = list()
    extentsLRBT.append(max(fromLRBT[0], toLRBT[0]))
    extentsLRBT.append(min(fromLRBT[1], toLRBT[1]))
    extentsLRBT.append(max(fromLRBT[2], toLRBT[2]))
    extentsLRBT.append(min(fromLRBT[3], toLRBT[3]))
    bbox = list()
    bbox.append([extentsLRBT[0], extentsLRBT[2]]) # Left Bottom
    bbox.append([extentsLRBT[1], extentsLRBT[2]]) # Right Bottom
    bbox.append([extentsLRBT[1], extentsLRBT[3]]) # Right Top
    bbox.append([extentsLRBT[0], extentsLRBT[3]]) # Left Top
    bpoly = geometry.Polygon(bbox)
    
    # crop from and to images to bounding box
    fromHeightCropped, t = rasterio.mask.mask(fromRaster, [bpoly], crop=True, indexes=3)
    fromErrorCropped, t = rasterio.mask.mask(fromRaster, [bpoly], crop=True, indexes=6)
    toHeightCropped, t = rasterio.mask.mask(toRaster, [bpoly], crop=True, indexes=3)
    toErrorCropped, t = rasterio.mask.mask(toRaster, [bpoly], crop=True, indexes=6)

    return fromHeightCropped, fromErrorCropped, toHeightCropped, toErrorCropped
    

def piv():
    # get image arrays of common (overlapping) area
    # we need the images to be the same size and cover the same geospatial extents
    fromHeight, fromError, toHeight, toError = get_arrays()
    print(fromHeight.shape)
    print(toHeight.shape)
    print(fromError.shape)
    print(toError.shape)

    # determine number of template window movements in horizontal (u) and vertical (v)

