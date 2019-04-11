import rasterio
import rasterio.plot
import rasterio.mask
from shapely import geometry
import numpy as np
import math
from skimage.feature import match_template
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import json

def get_image_arrays():    

    # read in the 'from' and 'to' images as numpy arrays (currently assumes multiple layers in the from and to image files)
    fromRaster = rasterio.open('from.tif')
    fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
    fromStd = fromRaster.read(6, masked=True)
    toRaster = rasterio.open('to.tif')
    toHeight = toRaster.read(3, masked=True)
    toStd = toRaster.read(6, masked=True) 
    
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
    fromHeightCropped, t = rasterio.mask.mask(fromRaster, [bpoly], crop=True, nodata=0, indexes=3)
    fromErrorCropped, t = rasterio.mask.mask(fromRaster, [bpoly], crop=True, nodata=0, indexes=6)
    toHeightCropped, t = rasterio.mask.mask(toRaster, [bpoly], crop=True, nodata=0, indexes=3)
    toErrorCropped, t = rasterio.mask.mask(toRaster, [bpoly], crop=True, nodata=0, indexes=6)

    return fromHeightCropped, fromErrorCropped, toHeightCropped, toErrorCropped

def subpx_peak_taylor(ncc):
    dx = (ncc[1,2] - ncc[1,0]) / 2
    dxx = ncc[1,2] + ncc[1,0] - 2*ncc[1,1]
    dy = (ncc[2,1] - ncc[0,1]) / 2
    dyy = ncc[2,1] + ncc[0,1] - 2*ncc[1,1]
    dxy = (ncc[2,2] - ncc[2,0] - ncc[0,2] + ncc[0,0]) / 4
    
    delX = -(dyy*dx - dxy*dy) / (dxx*dyy - dxy*dxy)
    delY = -(dxx*dy - dxy*dx) / (dxx*dyy - dxy*dxy)

    return [delX, delY] # delX is left-to-right; delY is top-to-bottom
    

def piv(templateSize, stepSize):

    # get image arrays of common (overlapping) area
    fromHeight, fromError, toHeight, toError = get_image_arrays()

    # determine number of search areas in horizontal (u) and vertical (v)
    searchSize = templateSize*2
    imageShape = fromHeight.shape # [height (rows), width (columns)]
    uCount = math.floor((imageShape[1]-searchSize) / stepSize)
    vCount = math.floor((imageShape[0]-searchSize) / stepSize)

    # cycle through each search area
    originUV = []
    offsetUV = []
    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for i in range(vCount):
        for j in range(uCount):

            # get template area data from the 'from' height and error images
            templateStartU = int(j*stepSize + math.ceil(templateSize/2))
            templateEndU = int(j*stepSize + math.ceil(templateSize/2) + templateSize)
            templateStartV = int(i*stepSize + math.ceil(templateSize/2))
            templateEndV = int(i*stepSize + math.ceil(templateSize/2) + templateSize)
            templateHeight = fromHeight[templateStartV:templateEndV, templateStartU:templateEndU].copy()
            templateError = fromError[templateStartV:templateEndV, templateStartU:templateEndU].copy()
            # get search area data from the 'to' height and error images
            searchStartU = int(j*stepSize)
            searchEndU = int(j*stepSize + searchSize + (templateSize % 2)) # the modulo addition forces the search area to be symmetric around odd-sized templates
            searchStartV = int(i*stepSize)
            searchEndV = int(i*stepSize + searchSize + (templateSize % 2)) # the modulo addition forces the search area to be symmetric around odd-sized templates
            searchHeight = toHeight[searchStartV:searchEndV, searchStartU:searchEndU].copy()
            searchError = toError[searchStartV:searchEndV, searchStartU:searchEndU].copy()            

            # move to next area if the template is flat, which breaks the correlation computation
            if ((templateHeight.max() - templateHeight.min()) == 0):
                continue
            
            # normalized cross correlation between the template and search area height data            
            ncc = match_template(searchHeight, templateHeight)
            nccMax = np.where(ncc == np.amax(ncc))

            # sub-pixel peak location
            if nccMax[0]==0 or nccMax[1]==0 or nccMax[0]==ncc.shape[0]-1 or nccMax[1]==ncc.shape[1]-1: # the subpixel interpolator can not handle peak locations on the edges of the correlation matrix
                subPxPeak = [0,0]
            else:
                subPxPeak = subpx_peak_taylor(ncc[nccMax[0].item(0)-1:nccMax[0].item(0)+2, nccMax[1].item(0)-1:nccMax[1].item(0)+2])
                      
            # store vector origin and end points            
            originUV.append(((j*stepSize + templateSize - (1 - templateSize % 2)*0.5), (i*stepSize + templateSize - (1 - templateSize % 2)*0.5))) # the expression containing the modulo operator adjusts even-sized template origins to be between pixel centers
            offsetUV.append(((nccMax[1] - math.ceil(templateSize/2) + subPxPeak[0]), (nccMax[0] - math.ceil(templateSize/2) + subPxPeak[1])))

            # show progress on 'from' and 'to' images
            plt.sca(ax1)
            plt.cla()
            ax1.set_title('FROM')
            ax1.imshow(fromHeight, cmap=plt.cm.gray)
            ax1.add_patch(Rectangle((templateStartU,templateStartV), templateSize-1, templateSize-1, linewidth=1, edgecolor='r',fill=None))
            plt.sca(ax2)
            plt.cla()
            ax2.set_title('TO')            
            ax2.imshow(toHeight, cmap=plt.cm.gray)            
            ax2.add_patch(Rectangle((searchStartU,searchStartV), searchSize-1, searchSize-1, linewidth=1, edgecolor='r',fill=None))
            plt.pause(0.01)


    # plot vectors on top of 'from' image
    plt.close(fig)
    fig, ax = plt.subplots()
    plt.imshow(fromHeight, cmap=plt.cm.gray)
    originUV = np.asarray(originUV)
    offsetUV = np.asarray(offsetUV)
    ax.quiver(originUV[...,0], originUV[:,1], offsetUV[:,0], offsetUV[:,1], angles='xy', color='r', linewidth=0)
    plt.show()

    # export vector origins and offsets to json file
    originUV = np.squeeze(originUV)
    offsetUV = np.squeeze(offsetUV)
    originOffset = np.concatenate((originUV, offsetUV), axis=1)
    jsonOut = originOffset.tolist()
    json.dump(jsonOut, open("piv_origins_offsets.json", "w"))
    print('PIV vector origins and offsets saved to file piv_origins_offsets.json')
