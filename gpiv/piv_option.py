import rasterio
import rasterio.plot
import rasterio.mask
from shapely import geometry
import numpy as np
import math
from skimage.feature import match_template
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import time
import json
from show_option import show
import time


def piv(templateSize, stepSize, propFlag):
    # set perturbation value for numeric partial derivatives
    if propFlag:
        p = 0.000001
        subPxPeakCov = []

    # get image arrays of common (overlapping) area
    fromHeight, fromError, toHeight, toError, transform = get_image_arrays(propFlag)

    # determine number of search areas in horizontal (u) and vertical (v)
    searchSize = templateSize*2
    imageShape = fromHeight.shape # [height (rows), width (columns)]
    uCount = math.floor((imageShape[1]-searchSize) / stepSize)
    vCount = math.floor((imageShape[0]-searchSize) / stepSize)

    # cycle through each set of search and template areas
    originUV = []
    offsetUV = []
    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for i in range(vCount):
        for j in range(uCount):
            # t00 = time.time()

            # get template area data from the 'from' height and error images
            templateStartU = int(j*stepSize + math.ceil(templateSize/2))
            templateEndU = int(j*stepSize + math.ceil(templateSize/2) + templateSize)
            templateStartV = int(i*stepSize + math.ceil(templateSize/2))
            templateEndV = int(i*stepSize + math.ceil(templateSize/2) + templateSize)
            templateHeight = fromHeight[templateStartV:templateEndV, templateStartU:templateEndU].copy()
            
            # get search area data from the 'to' height and error images
            searchStartU = int(j*stepSize)
            searchEndU = int(j*stepSize + searchSize + (templateSize % 2)) # the modulo addition forces the search area to be symmetric around odd-sized templates
            searchStartV = int(i*stepSize)
            searchEndV = int(i*stepSize + searchSize + (templateSize % 2)) # the modulo addition forces the search area to be symmetric around odd-sized templates
            searchHeight = toHeight[searchStartV:searchEndV, searchStartU:searchEndU].copy()            

            # show template and search patches on 'from' and 'to' images for visual progress indication
            plt.sca(ax1)
            plt.cla()
            ax1.set_title('FROM')
            ax1.imshow(fromHeight, cmap=plt.cm.gray)
            ax1.add_patch(pch.Rectangle((templateStartU,templateStartV), templateSize-1, templateSize-1, linewidth=1, edgecolor='r',fill=None))
            plt.sca(ax2)
            plt.cla()
            ax2.set_title('TO')            
            ax2.imshow(toHeight, cmap=plt.cm.gray)            
            ax2.add_patch(pch.Rectangle((searchStartU,searchStartV), searchSize-1, searchSize-1, linewidth=1, edgecolor='r',fill=None))
            plt.pause(0.1)        

            # move to next area if the template is flat, which breaks the correlation computation
            if ((templateHeight.max() - templateHeight.min()) == 0):
                continue
            
            # normalized cross correlation between the template and search area height data - fast, but based on FFT          
            ncc = match_template(searchHeight, templateHeight)
            # normalized cross correlation between the template and search area height data - slower, but is pure spatial ncc (not FFT-based) 
            # ncc = ncc_running_sums(searchHeight, templateHeight)

            # maximum in the ncc surface
            nccMax = np.where(ncc == np.amax(ncc))

            # sub-pixel peak location
            if nccMax[0][0]==0 or nccMax[1][0]==0 or nccMax[0][0]==ncc.shape[0]-1 or nccMax[1][0]==ncc.shape[1]-1: # the subpixel interpolator can not handle peak locations on the edges of the correlation matrix
                continue
            else:
                print(ncc[nccMax[0][0]-1:nccMax[0][0]+2, nccMax[1][0]-1:nccMax[1][0]+2])
                subPxPeak = subpx_peak_taylor(ncc[nccMax[0][0]-1:nccMax[0][0]+2, nccMax[1][0]-1:nccMax[1][0]+2])
                    
            # store vector origin and end points            
            originUV.append(((j*stepSize + templateSize - (1 - templateSize % 2)*0.5), (i*stepSize + templateSize - (1 - templateSize % 2)*0.5))) # the expression containing the modulo operator adjusts even-sized template origins to be between pixel centers
            offsetUV.append(((nccMax[1][0] - math.ceil(templateSize/2) + subPxPeak[0]), (nccMax[0][0] - math.ceil(templateSize/2) + subPxPeak[1])))

            # propagate error if requested
            if propFlag:
                # get template and search areas from the 'from' and 'to' error images
                templateError = fromError[templateStartV:templateEndV, templateStartU:templateEndU].copy()
                searchError = toError[searchStartV:searchEndV, searchStartU:searchEndU].copy()    

                # propagate raster error into the 3x3 patch of correlation values that are centered on the correlation peak
                # t0 = time.time()
                nccCov = prop_px2corr(templateHeight,
                                      templateError, 
                                      searchHeight[nccMax[0][0]-1:nccMax[0][0]+templateSize+1, nccMax[1][0]-1:nccMax[1][0]+templateSize+1], # templateSize+2 x templateSize+2 subarray of the search array,
                                      searchError[nccMax[0][0]-1:nccMax[0][0]+templateSize+1, nccMax[1][0]-1:nccMax[1][0]+templateSize+1], # templateSize+2 x templateSize+2 subarray of the search error array
                                      ncc[nccMax[0][0]-1:nccMax[0][0]+2, nccMax[1][0]-1:nccMax[1][0]+2], # 3x3 array of correlation values centered on the correlation peak
                                      p) 
                # t1 = time.time()
                # print("prop time={}".format(t1-t0))

                # propagate the correlation covariance into the sub-pixel peak location
                peakCov = prop_corr2peak(ncc[nccMax[0][0]-1:nccMax[0][0]+2, nccMax[1][0]-1:nccMax[1][0]+2],
                                         nccCov,
                                         subPxPeak,
                                         p)

                # convert covariance matrix from pixels squared to ground distance squared
                peakCov[0][0] *= transform[0]*transform[0]
                peakCov[0][1] *= transform[0]*transform[0]
                peakCov[1][0] *= transform[0]*transform[0]
                peakCov[1][1] *= transform[0]*transform[0]

                # store for json output; we convert to a list here for simple json output
                subPxPeakCov.append(peakCov.tolist())   

            # t11 = time.time()
            # print("total time={}".format(t11-t00))


    # convert vector origins and offsets from pixels to ground distance json file
    originUV = np.asarray(originUV)        
    originUV *= transform[0] # scale by pixel ground size
    originUV[:,0] += transform[2] # offset U by leftmost pixel to get ground coordinate
    originUV[:,1] = transform[5] - originUV[:,1] # subtract V from uppermost pixel to get ground coordinate
    offsetUV = np.asarray(offsetUV)
    offsetUV *= transform[0] # scale by pixel ground size

    # export vector origins and offsets to json
    originOffset = np.concatenate((originUV, offsetUV), axis=1)
    # jsonOut = originOffset.tolist()
    json.dump(originOffset.tolist(), open("piv_origins_offsets.json", "w"))
    print("PIV vector origins and offsets saved to file 'piv_origins_offsets.json'")

    # export covariance matrices to json
    if propFlag:
        json.dump(subPxPeakCov, open("piv_covariance_matrices.json", "w"))
        print("PIV displacement covariance matrices saved to file 'piv_covariance_matrices.json'")

    # plot the displacement vectors on top of 'from' image 
    plt.close(fig)
    maxImgDim = np.amax(fromHeight.shape) * transform[0]
    if propFlag:        
        vecSF, ellSF = vec_ellipse_scales(offsetUV, subPxPeakCov, maxImgDim, propFlag)
        show(True, False, True, False, True, vecSF, True, ellSF)
    else:
        vecSF, ellSF = vec_ellipse_scales(offsetUV, [], maxImgDim, propFlag)
        show(True, False, True, False, True, vecSF, False, False)


def vec_ellipse_scales(offsetUV, subPxPeakCov, maxImgDim, propFlag):
    # Original code deleted. Now handled by the "show" module, which does a better job
    # The ellipse scale factor will also need to be moved to the "show" module
    vecSF = 1

    # generate ellipse scale factor such that the median semimajor axis magnitude is 1/45 the maximum image dimension
    if propFlag:
        semimajor = []
        for i in range(len(subPxPeakCov)):
            eigenVals, eigenVecs = np.linalg.eig(subPxPeakCov[i])
            idxMax = np.argmax(eigenVals)
            semimajor.append(math.sqrt(2.298*eigenVals[idxMax])) # scale factor of 2.298 to create a 68% confidence ellipse
        medianSemimajor = np.median(semimajor)
        ellSF = (maxImgDim / 45) / medianSemimajor
    else:
        ellSF = []

    return vecSF, ellSF


def prop_px2corr(template, templateError, search, searchError, ncc, p):
    # form diagonal covariance matrix from template and search patch covariance arrays
    templateCovVec = np.square(templateError.reshape(templateError.size,)) # convert array to vector, row-by-row, and square the standard deviations into variances
    searchCovVec = np.square(searchError.reshape(searchError.size,))
    covVec = np.hstack((templateCovVec, searchCovVec))
    C = np.diag(covVec)

    # get the Jacobian
    J = ncc_jacobian(template, search, ncc, p)

    # propagate the template and search area errors into the 9 correlation elements; the covariance order is by row of the ncc array (i.e., ncc[0,0], ncc[0,1], ncc[0,2], ncc[1,0], ncc[1,1], ...)
    nccCov = np.matmul(np.matmul(J,C),J.T)

    return nccCov


def ncc_jacobian(template, search, ncc, p):
    # define some loop sizes and pre-allocate the Jacobian
    tRow, tCol = template.shape
    sRow, sCol = search.shape
    jacobian = np.zeros((9, template.size + search.size))

    sz = template.size
    templateN = (template - np.mean(template)) / (np.std(template))

    # cycle through the 3x3 correlation array, row-by-row
    for i in range(3): # rows
        for j in range(3): # columns
            # pull out the sub-area of the search patch
            searchSub = search[i:i+tRow, j:j+tCol]

            searchSubN = (searchSub - np.mean(searchSub)) / (np.std(searchSub))
            
            # preallocate arrays to store the template and search partial derivates
            templatePartials = np.zeros((tRow, tCol))
            searchPartials = np.zeros((sRow, sCol))

            # now cycle through each template and sub-search area pixel and numerically estimate its partial derivate with respect to the normalized cross correlation
            for m in range(tRow):
                for n in range(tCol):
                    # perturb
                    # print(type(template[0][0]))
                    templatePerturb = template.copy() # we make a copy here because we will modify an element (do not want that modification to change the original value)
                    templatePerturb[m,n] += p
                    searchSubPerturb = searchSub.copy()
                    searchSubPerturb[m,n] += p

                    # compute perturbed ncc - prior method that is much slower
                    # nccTemplatePerturb = match_template(templatePerturb, searchSub)
                    # nccSearchPerturb = match_template(template, searchSubPerturb)

                    # Potential Method that may catch the NaN generation
                    # templatePerturbStd = np.std(templatePerturb)
                    # searchSubPerturbStd = np.std(searchSubPerturb)
                    # if templatePerturbStd:
                    #     templatePerturbN = (templatePerturb - np.mean(templatePerturb)) / (templatePerturbStd * sz)
                    # else:
                    #     templatePerturbN = 0
                    # if searchSubPerturbStd:
                    #     searchSubPerturbN = (searchSubPerturb - np.mean(searchSubPerturb)) / (searchSubPerturbStd * sz)
                    # else:
                    #     searchSubPerturbN = 0

                    # compute perturbed ncc - brute force method that is about 20x faster than using skimage's match_template
                    templatePerturbN = (templatePerturb - np.mean(templatePerturb)) / (np.std(templatePerturb) * sz)
                    searchSubPerturbN = (searchSubPerturb - np.mean(searchSubPerturb)) / (np.std(searchSubPerturb) * sz)
                    nccTemplatePerturb = np.sum(templatePerturbN * searchSubN)
                    nccSearchPerturb = np.sum(templateN * searchSubPerturbN)
                    # print(type(nccSearchPerturb))
                    # print(type(ncc[i,j]))
                    # print(nccSearchPerturb)
                    # print(ncc[i,j])
                    # print(nccSearchPerturb - ncc[i,j])
                    
                    # numeric partial derivatives
                    templatePartials[m,n] = (nccTemplatePerturb - ncc[i,j]) / p
                    searchPartials[i+m,j+n] = (nccSearchPerturb - ncc[i,j]) / p # the location adjustment by i and j accounts for the larger size of the search area than the template area
                    # print((nccSearchPerturb - ncc[i,j])/p)

            # reshape the partial derivatives from their current array form to vector form and store in the Jacobian; note that we match the row-by-row pattern used to form the covariance matrix in the calling function
            jacobian[i*3+j, 0:template.size] = templatePartials.reshape(templatePartials.size,)
            jacobian[i*3+j, template.size:template.size+search.size] = searchPartials.reshape(searchPartials.size,)
    
    # np.set_printoptions(precision=3, suppress=True)
    # print(jacobian)
    # print(jacobian.shape)
    return jacobian


def prop_corr2peak(ncc, nccCov, deltaUV, p):
    # pre-allocate Jacobian
    jacobian = np.zeros((2,9))

    # cycle through the 3x3 correlation array, row-by-row, and create the jacobian matrix
    for i in range(3): # rows
        for j in range(3): # columns
            nccPerturb = ncc.copy()
            nccPerturb[i,j] += p
            deltaUPerturb, deltaVPerturb = subpx_peak_taylor(nccPerturb)
            jacobian[0,i*3+j] = (deltaUPerturb - deltaUV[0]) / p
            jacobian[1,i*3+j] = (deltaVPerturb - deltaUV[1]) / p
            # print(deltaVPerturb - deltaUV[1])
    
    # propagate the 3x3 array of correlation uncertainties into the sub-pixel U and V direction offsets
    subPxPeakCov = np.matmul(np.matmul(jacobian,nccCov),jacobian.T)
            
    return subPxPeakCov


def get_image_arrays(propFlag):    
    # read in the 'from' and 'to' height images as numpy arrays 
    fromHeight = rasterio.open('fromHeight.tif')
    toHeight = rasterio.open('toHeight.tif')
    
    # if propagating error, read in the 'from' and 'to' error images as numpy arrays
    if propFlag:
        fromError = rasterio.open('fromError.tif')
        toError = rasterio.open('toError.tif')

    # get the raster geometric transformation - assumes from and to height images have same transform
    transform = fromHeight.transform
    
    # create a polygon defining the extents of the geospatial overlap
    fromLRBT = list(rasterio.plot.plotting_extent(fromHeight)) # LRBT = [left, right, bottom, top]
    toLRBT = list(rasterio.plot.plotting_extent(toHeight))
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
    fromHeightCropped, t = rasterio.mask.mask(fromHeight, [bpoly], crop=True, nodata=0, indexes=1)
    fromHeightCropped = fromHeightCropped.astype('float64')
    toHeightCropped, t = rasterio.mask.mask(toHeight, [bpoly], crop=True, nodata=0, indexes=1)
    toHeightCropped = toHeightCropped.astype('float64')

    if propFlag:
        fromErrorCropped, t = rasterio.mask.mask(fromError, [bpoly], crop=True, nodata=0, indexes=1)
        fromErrorCropped = fromHeightCropped.astype('float64')
        toErrorCropped, t = rasterio.mask.mask(toError, [bpoly], crop=True, nodata=0, indexes=1)
        toErrorCropped = toErrorCropped.astype('float64')
    else:
        fromErrorCropped = []
        toErrorCropped = []

    return fromHeightCropped, fromErrorCropped, toHeightCropped, toErrorCropped, transform


def subpx_peak_taylor(ncc):
    dx = (ncc[1,2] - ncc[1,0]) / 2
    dxx = ncc[1,2] + ncc[1,0] - 2*ncc[1,1]
    dy = (ncc[2,1] - ncc[0,1]) / 2
    dyy = ncc[2,1] + ncc[0,1] - 2*ncc[1,1]
    dxy = (ncc[2,2] - ncc[2,0] - ncc[0,2] + ncc[0,0]) / 4
    
    delX = -(dyy*dx - dxy*dy) / (dxx*dyy - dxy*dxy)
    delY = -(dxx*dy - dxy*dx) / (dxx*dyy - dxy*dxy)

    return [delX, delY] # delX is left-to-right; delY is top-to-bottom; note that delX == delU and delY == delV


def ncc_running_sums(search, template):
    s = np.zeros((search.shape[0]+1, search.shape[1]+1))
    s2 = s.copy()
    for u in range(1,search.shape[1]+1): # columns
        for v in range(1,search.shape[0]+1): # rows
            s[u,v] = search[u-1,v-1] + s[u-1,v] + s[u,v-1] - s[u-1,v-1]
            s2[u,v] = search[u-1,v-1]**2 + s2[u-1,v] + s2[u,v-1] - s2[u-1,v-1]
    
    templateZeroMean = template  - np.mean(template)
    templateZeroMeanSquareSum = np.sum(templateZeroMean**2)
    N = template.shape[0]
    M = search.shape[0]

    ncc = np.zeros((M-N+1,M-N+1))
    for u in range(M-N+1):
        for v in range (M-N+1):
            searchSum = s[u+N,v+N] - s[u,v+N] - s[u+N,v] + s[u,v]
            numerator = np.sum(search[u:u+N,v:v+N] * templateZeroMean)
            denominator = np.sqrt(((s2[u+N,v+N] - s2[u,v+N] - s2[u+N,v] + s2[u,v]) - searchSum**2 / (N*N)) * templateZeroMeanSquareSum)
            ncc[u,v] = numerator / denominator
    
    return ncc
