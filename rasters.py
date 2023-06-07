# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:27:55 2023

This ode contains a function that is used to save a raster file into geotiff format.

@author: nekhtari@uh.edu
"""



from osgeo import gdal, osr


def write_raster(outname, ul, pixel_width, pixel_height, array, epsg):
    
    '''
    input:
        outname:      String defining path + filename of the output raster
        ul:           Coordinates of upper left corner of the raster to be saved
        pixel_width:  Ground pixel size in east-west direction
        pixel_height: Ground pixel size in north-south direction
        array:        The input raster we are writing to disk
        epsg:         The 4-5 digit EPSG code specifying projection system and datums
    output:
        None
    '''

    cols = array.shape[1]
    rows = array.shape[0]
    originX = ul[0]
    originY = ul[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(outname, cols, rows, 1, gdal.GDT_Float64)
    #outRaster = driver.Create(outname, cols, rows, 3, gdal.GDT_Byte)
    
    outRaster.SetGeoTransform((originX, pixel_width, 0, originY, 0, pixel_height))
    
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outRaster = None






