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


class Piv:

    def __init__(self):
        # lists to store numpy arrays of data necessary for the optional error propagation
        self.templateStore = []
        self.templateErrorStore = []
        self.searchStore = []
        self.searchErrorStore = []
        self.correlationStore = []
        self.subPxUvStore = []
        self.searchBoxStore = []
        self.templateBoxStore = []
        self.toHeightStore = np.array([])
        self.fromHeightStore = np.array([])
        self.originUvStore = []
        self.offsetUvStore = []
        # HARD-CODED perturbation value for generating numeric partial derivatives
        self.p = 0.00001


    def compute(self, templateSize, stepSize, propFlag):
        # get image arrays of common (overlapping) area
        fromHeight, fromError, toHeight, toError = self._get_image_arrays()

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
                if nccMax[0][0]==0 or nccMax[1][0]==0 or nccMax[0][0]==ncc.shape[0]-1 or nccMax[1][0]==ncc.shape[1]-1: # the subpixel interpolator can not handle peak locations on the edges of the correlation matrix
                    continue
                else:
                    subPxPeak = self._subpx_peak_taylor(ncc[nccMax[0][0]-1:nccMax[0][0]+2, nccMax[1][0]-1:nccMax[1][0]+2])
                        
                # store vector origin and end points            
                originUV.append(((j*stepSize + templateSize - (1 - templateSize % 2)*0.5), (i*stepSize + templateSize - (1 - templateSize % 2)*0.5))) # the expression containing the modulo operator adjusts even-sized template origins to be between pixel centers
                offsetUV.append(((nccMax[1][0] - math.ceil(templateSize/2) + subPxPeak[0]), (nccMax[0][0] - math.ceil(templateSize/2) + subPxPeak[1])))

                # store data for error propagation if necessary
                if propFlag:
                    # data for propagation computations
                    self.templateStore.append(templateHeight)
                    self.templateErrorStore.append(templateError)
                    self.searchStore.append(searchHeight[nccMax[0][0]-1:nccMax[0][0]+templateSize+1, nccMax[1][0]-1:nccMax[1][0]+templateSize+1]) # templateSize+2 x templateSize+2 subarray of the search array
                    self.searchErrorStore.append(searchError[nccMax[0][0]-1:nccMax[0][0]+templateSize+1, nccMax[1][0]-1:nccMax[1][0]+templateSize+1]) # templateSize+2 x templateSize+2 subarray of the search error array
                    self.correlationStore.append(ncc[nccMax[0][0]-1:nccMax[0][0]+2, nccMax[1][0]-1:nccMax[1][0]+2]) # 3x3 array of correlation values centered on the correlation peak
                    self.subPxUvStore.append(subPxPeak)
                    # data for status plot
                    self.searchBoxStore.append([searchStartU, searchEndV, searchSize-1, searchSize-1]) # [left, bottom, width, height]; the '-1' operations adjust the box so it is plotted correctly
                    self.templateBoxStore.append([searchStartU+nccMax[1][0], searchEndV+nccMax[0][0], templateSize-1, templateSize-1])                    

                # show progress on 'from' and 'to' images
                plt.sca(ax1)
                plt.cla()
                ax1.set_title('FROM')
                ax1.imshow(fromHeight, cmap=plt.cm.gray)
                ax1.add_patch(pch.Rectangle((templateStartU,templateEndV), templateSize-1, templateSize-1, linewidth=1, edgecolor='r',fill=None))
                plt.sca(ax2)
                plt.cla()
                ax2.set_title('TO')            
                ax2.imshow(toHeight, cmap=plt.cm.gray)            
                ax2.add_patch(pch.Rectangle((searchStartU,searchEndV), searchSize-1, searchSize-1, linewidth=1, edgecolor='r',fill=None))
                plt.pause(0.01)

        # export vector origins and offsets to json file
        originUV = np.squeeze(originUV)
        offsetUV = np.squeeze(offsetUV)
        originOffset = np.concatenate((originUV, offsetUV), axis=1)
        jsonOut = originOffset.tolist()
        json.dump(jsonOut, open("piv_origins_offsets.json", "w"))
        print('PIV vector origins and offsets saved to file piv_origins_offsets.json')

        # wrap it up with a plot vectors on top of 'from' image or by storing the toHeight image for later use
        plt.close(fig)
        if propFlag:
            # data for status and final error propagation plots 
            self.fromHeightStore = fromHeight
            self.toHeightStore = toHeight
            self.originUvStore = originUV
            self.offsetUvStore = offsetUV
        else: # only show vectors if we are not moving on to propagate errors
            fig, ax = plt.subplots()
            plt.imshow(fromHeight, cmap=plt.cm.gray)
            originUV = np.asarray(originUV)
            offsetUV = np.asarray(offsetUV)
            ax.quiver(originUV[:,0], originUV[:,1], offsetUV[:,0], offsetUV[:,1], angles='xy', color='r', linewidth=0)
            plt.show()


    def propagate(self):
        # cycle through each set of stored data
        subPxPeakCov = []
        fig, ax = plt.subplots()
        for i in range(len(self.templateStore)):
            # show progress on the 'to' image                        
            plt.cla()
            ax.set_title("Error Propagation Status: Computation Location")
            ax.imshow(self.toHeightStore, cmap=plt.cm.gray)
            # ax.add_patch(pch.Rectangle((self.searchBoxStore[i][0], self.searchBoxStore[i][1]), self.searchBoxStore[i][2], self.searchBoxStore[i][3], linewidth=1, edgecolor='r',fill=None))
            ax.add_patch(pch.Rectangle((self.templateBoxStore[i][0], self.templateBoxStore[i][1]), self.templateBoxStore[i][2], self.templateBoxStore[i][3], linewidth=1, edgecolor='r',fill=None))
            plt.pause(0.1)

            # propagate raster error into the 3x3 patch of correlation values that are centered on the correlation peak
            nccCov = self._prop_px2corr(self.templateStore[i], self.templateErrorStore[i], self.searchStore[i], self.searchErrorStore[i], self.correlationStore[i])

            # propagate the correlation covariance into the sub-pixel peak location
            peakCov = self._prop_corr2peak(self.correlationStore[i], nccCov, self.subPxUvStore[i])

            # store for plotting and json output; we convert to a list here for simple json output
            subPxPeakCov.append(peakCov.tolist())

        # close status figure and save peak location covariance matrices to json file
        plt.close(fig)
        json.dump(subPxPeakCov, open("piv_covariance_matrices.json", "w"))
        print('PIV displacement covariance matrices saved to file piv_covariance_matrices.json')

        # plot displacement vectors and 68% confidence (1-sigma) error ellipses on top of 'from' image
        # the computed vectors will be scaled such that the median vector displacement magnitude is 1/30 the maximum image dimension
        maxImgDim = np.amax(self.fromHeightStore.shape)
        vecNorms = np.linalg.norm(self.offsetUvStore, axis=1)
        medianVecNorm = np.median(vecNorms)
        vecScale = (maxImgDim / 30) / medianVecNorm
        vecHeadSize = medianVecNorm*vecScale / 3
        # the computed ellipses will be scaled such that the median semimajor axis magnitude is 1/60 the maximum image dimension
        semimajor = []
        semiminor = []
        angle = []
        for i in range(len(self.originUvStore)):
            eigenVals, eigenVecs = np.linalg.eig(subPxPeakCov[i])
            idxMax = np.argmax(eigenVals)
            idxMin = np.argmin(eigenVals)
            semimajor.append(math.sqrt(2.298*eigenVals[idxMax])) # scale factor of 2.298 to create a 68% confidence ellipse
            semiminor.append(math.sqrt(2.298*eigenVals[idxMin]))
            angle.append(np.degrees(np.arctan(eigenVecs[idxMax][1]/eigenVecs[idxMax][0])))
        medianSemimajor = np.median(semimajor)
        ellScale = (maxImgDim / 45) / medianSemimajor

        fig, ax = plt.subplots()
        plt.imshow(self.fromHeightStore, cmap=plt.cm.gray)
        for i in range(len(self.originUvStore)):
            # add arrow
            a = pch.FancyArrow(self.originUvStore[i][0], self.originUvStore[i][1], 
                               self.offsetUvStore[i][0]*vecScale, self.offsetUvStore[i][1]*vecScale, 
                               length_includes_head=True, head_width=vecHeadSize, head_length=vecHeadSize, overhang=0.8, fc='red', ec='red')
            ax.add_artist(a)
            # add ellipse centered at location of actual (not scaled) displacement            
            e = pch.Ellipse((self.originUvStore[i][0]+self.offsetUvStore[i][0], self.originUvStore[i][1]+self.offsetUvStore[i][1]), 
                            semimajor[i]*ellScale, semiminor[i]*ellScale, angle=angle[i])            
            ax.add_artist(e)
    
        plt.show()

    
    def _prop_px2corr(self, template, templateError, search, searchError, ncc):
        # form diagonal covariance matrix from template and search patch covariance arrays
        templateCovVec = templateError.reshape(templateError.size,) # convert array to vector, row-by-row
        searchCovVec = searchError.reshape(searchError.size,)
        covVec = np.hstack((templateCovVec, searchCovVec))
        C = np.diag(covVec)

        # get the Jacobian
        J = self._ncc_jacobian(template, search, ncc)
        # print('Jacobian size = %u' % J.size)

        # propagate the template and search area errors into the 9 correlation elements; the covariance order is by row of the ncc array (i.e., ncc[0,0], ncc[0,1], ncc[0,2], ncc[1,0], ncc[1,1], ...)
        nccCov = np.matmul(np.matmul(J,C),J.T)

        return nccCov


    def _ncc_jacobian(self, template, search, ncc):
        # define some loop sizes and pre-allocate the Jacobian
        tRow, tCol = template.shape
        # print(tRow)
        # print(tCol)
        sRow, sCol = search.shape
        # print(sRow)
        # print(sCol)
        jacobian = np.zeros((9, template.size + search.size))

        # cycle through the 3x3 correlation array, row-by-row
        for i in range(3): # rows
            for j in range(3): # columns
                # pull out the sub-area of the search patch
                searchSub = search[i:i+tRow, j:j+tCol]
                # print(searchSub.size)

                # preallocate arrays to store the template and search partial derivates
                templatePartials = np.zeros((tRow, tCol))
                searchPartials = np.zeros((sRow, sCol))

                # now cycle through each template and sub-search area pixel and numerically estimate its partial derivate with respect to the normalized cross correlation
                for m in range(tRow):
                    for n in range(tCol):
                        # perturb
                        templatePerturb = template.copy() # we make a copy here because we will modify an element (do not want that modification to change the original value)
                        templatePerturb[m,n] += self.p
                        searchSubPerturb = searchSub.copy()
                        searchSubPerturb[m,n] += self.p

                        # compute perturbed ncc
                        nccTemplatePerturb = match_template(templatePerturb, searchSub)
                        nccSearchPerturb = match_template(template, searchSubPerturb)
                        
                        # numeric partial derivatives
                        templatePartials[m,n] = (nccTemplatePerturb - ncc[i,j]) / (self.p)
                        searchPartials[i+m,j+n] = (nccSearchPerturb - ncc[i,j]) / (self.p) # the location adjustment by i and j accounts for the larger size of the search area than the template area

                # reshape the partial derivatives from their current array form to vector form and store in the Jacobian; note that we match the row-by-row pattern used to form the covariance matrix in the calling function
                jacobian[i*3+j, 0:template.size] = templatePartials.reshape(templatePartials.size,)
                jacobian[i*3+j, template.size:template.size+search.size] = searchPartials.reshape(searchPartials.size,)
        
        return jacobian


    def _prop_corr2peak(self, ncc, nccCov, deltaUV):
        # pre-allocate Jacobian
        jacobian = np.zeros((2,9))

        # cycle through the 3x3 correlation array, row-by-row, and create the jacobian matrix
        for i in range(3): # rows
            for j in range(3): # columns
                nccPerturb = ncc.copy()
                nccPerturb[i,j] += self.p
                deltaUPerturb, deltaVPerturb = self._subpx_peak_taylor(nccPerturb)
                jacobian[0,i*3+j] = (deltaUPerturb - deltaUV[0]) / (self.p)
                jacobian[1,i*3+j] = (deltaVPerturb - deltaUV[1]) / (self.p)
        
        # propagate the 3x3 array of correlation uncertainties into the sub-pixel U and V direction offsets
        subPxPeakCov = np.matmul(np.matmul(jacobian,nccCov),jacobian.T)
                
        return subPxPeakCov


    def _get_image_arrays(self):    
        # read in the 'from' and 'to' images as numpy arrays (currently assumes multiple layers in the from and to image files)
        fromRaster = rasterio.open('from.tif')
        toRaster = rasterio.open('to.tif')
        
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


    def _subpx_peak_taylor(self, ncc):
        dx = (ncc[1,2] - ncc[1,0]) / 2
        dxx = ncc[1,2] + ncc[1,0] - 2*ncc[1,1]
        dy = (ncc[2,1] - ncc[0,1]) / 2
        dyy = ncc[2,1] + ncc[0,1] - 2*ncc[1,1]
        dxy = (ncc[2,2] - ncc[2,0] - ncc[0,2] + ncc[0,0]) / 4
        
        delX = -(dyy*dx - dxy*dy) / (dxx*dyy - dxy*dxy)
        delY = -(dxx*dy - dxy*dx) / (dxx*dyy - dxy*dxy)

        return [delX, delY] # delX is left-to-right; delY is top-to-bottom; note that delX == delU and delY == delV
    


