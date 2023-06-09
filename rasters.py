# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:27:55 2023

This ode contains a function that is used to save a raster file into geotiff format.

reference doc:
    rasterio documentation:
        https://buildmedia.readthedocs.org/media/pdf/rasterio/stable/rasterio.pdf


@author: nekhtari@uh.edu
"""



import rasterio
from rasterio.transform import Affine


def write_raster(outname, array, ul, pixel_width, pixel_height, epsg):
    
    width, height, num_bands = get_dims(array)
    crs = rasterio.crs.CRS.from_epsg(epsg)
    transform = Affine.translation(ul[0], ul[1]) * Affine.scale(pixel_width, -pixel_height)
    
        # Set raster profile
    profile = {
    'driver': 'GTiff',
    'dtype': array.dtype,
    'nodata': 9999,
    'width': width,
    'height':height,
    'count': num_bands,
    'crs': crs,
    'transform': transform,
    'tiled': True,
    'compress': 'lzw'
    }
    
    
    with rasterio.open(outname, 'w', **profile) as dst:
        dst.write(array, 1)
    
    


def get_dims(array):
    sh = array.shape
    width  = sh[1]
    height = sh[0]
    nbands = sh[2] if len(sh) == 3 else 1
    return (width, height, nbands)






