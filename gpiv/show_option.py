import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
import numpy as np
import json
import matplotlib.patches as patch
import math


def show(f, t, h, e, vec, vecSF, ell, ellSF):
    # open the requested raster and plot
    ax, tm = plot_raster(f, t, h, e)

    # open vectors and plot if requested
    if vec:
        vecSF = float(vecSF)
        plot_vectors(ax, tm, vecSF)

    # open ellipses and plot if requested
    if ell:
        ellSF = float(ellSF)
        plot_ellipses(ax, ellSF)

    # show
    plt.show()


def plot_vectors(ax, tm, vecSF):
    with open('piv_origins_offsets.json') as jsonFile:
        d = json.load(jsonFile)
    
    # nominal vector lenght scale factor
    dn = np.asarray(d)
    vecLen = np.linalg.norm(dn[:,2:], axis=1) / tm[0] # convert to pixels
    nomVecSF = 5/np.median(vecLen) # scale factor to convert median vector length to 5 pixels
        
    for i in range(len(d)):
        # add arrow with base at vector origin
        a = patch.FancyArrow(d[i][0], d[i][1], d[i][2]*nomVecSF*vecSF, -d[i][3]*nomVecSF*vecSF, # the negative sign converts from dV (postive down) to dY (positive up)
                            length_includes_head=True, head_width=vecSF*1, head_length=vecSF*1, overhang=0.8, fc='green', ec='green')
        ax.add_artist(a)


def plot_ellipses(ax, ellSF):
    with open('piv_origins_offsets.json') as jsonFile:
        d = json.load(jsonFile)
    with open('piv_covariance_matrices.json') as jsonFile:
        c = json.load(jsonFile)
    
    for i in range(len(d)):
        # semi-major and minor axes directions and half-lengths
        eigenVals, eigenVecs = np.linalg.eig(c[i])
        idxMax = np.argmax(eigenVals)
        idxMin = np.argmin(eigenVals)
        semimajor = math.sqrt(2.298*eigenVals[idxMax]) # scale factor of 2.298 to create a 68% confidence ellipse
        semiminor = math.sqrt(2.298*eigenVals[idxMin])
        angle = np.degrees(np.arctan(eigenVecs[idxMax][1]/eigenVecs[idxMax][0]))
        # add ellipse centered at location of actual (not scaled) displacement            
        e = patch.Ellipse((d[i][0]+d[i][2], d[i][1]-d[i][3]), semimajor*ellSF, semiminor*ellSF, # the negative sign in 'd[i][1]-d[i][3]' converts from dV (postive down) to dY (positive up)
                          angle=angle, fc='None', ec='red')            
        ax.add_artist(e)


def plot_raster(f, t, h, e):
    if f:
        if h:
            with rasterio.open('fromHeight.tif') as src:
                plotRaster = src.read(1, masked=True)
                plotLRBT = list(rasterio.plot.plotting_extent(src)) # LRBT = [left, right, bottom, top]
            plotTitle = 'From (height)'
        else:
            with rasterio.open('fromError.tif') as src:
                plotRaster = src.read(1, masked=True)
                plotLRBT = list(rasterio.plot.plotting_extent(src)) 
            plotTitle = 'From (error)'
    else:
        if h:
            with rasterio.open('toHeight.tif') as src:
                plotRaster = src.read(1, masked=True)
                plotLRBT = list(rasterio.plot.plotting_extent(src)) 
            plotTitle = 'To (height)'
        else:
            with rasterio.open('toError.tif') as src:
                plotRaster = src.read(1, masked=True)
                plotLRBT = list(rasterio.plot.plotting_extent(src)) 
            plotTitle = 'To (error)'

    tm = src.transform
    src.close()
    plotMin = min(np.percentile(plotRaster.compressed(), 1), np.percentile(plotRaster.compressed(), 1))
    plotMax = max(np.percentile(plotRaster.compressed(), 99), np.percentile(plotRaster.compressed(), 99))
    ax = plt.gca()
    plt.imshow(plotRaster, cmap=plt.cm.gray, extent=plotLRBT,vmin=plotMin,vmax=plotMax)
    ax.set_title(plotTitle)

    return ax, tm








    # old code that can be partially re-used if we add an option to show both 'from' and 'to' rasters simultaneously
    # # get the data
    # fromRaster = rasterio.open('from.tif')
    # fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
    # fromStd = fromRaster.read(6, masked=True)

    # toRaster = rasterio.open('to.tif')
    # toHeight = toRaster.read(3, masked=True)
    # toStd = toRaster.read(6, masked=True) 
        
    # # Determine common bounds for 'to' and 'from' raster spatial extents and
    # # values so that differences in extents and values are apparent when
    # # plotted.
    # # 1. Determine bounding box encompassing both the 'from' and 'to' rasters
    # fromLRBT = list(rasterio.plot.plotting_extent(fromRaster)) # LRBT = [left, right, bottom, top]
    # toLRBT = list(rasterio.plot.plotting_extent(toRaster))
    # LRBT = list()
    # LRBT.append(min(fromLRBT[0], toLRBT[0]))
    # LRBT.append(max(fromLRBT[1], toLRBT[1]))
    # LRBT.append(min(fromLRBT[2], toLRBT[2]))
    # LRBT.append(max(fromLRBT[3], toLRBT[3]))
    # # 2. Get maximum and minimum array values for both the 'from' and 'to' rasters. Using percentiles to avoid outliers.
    # minHeight = min(np.percentile(fromHeight.compressed(), 1), np.percentile(toHeight.compressed(), 1))
    # maxHeight = max(np.percentile(fromHeight.compressed(), 99), np.percentile(toHeight.compressed(), 99))
    # minStd = min(np.percentile(fromStd.compressed(), 1), np.percentile(toStd.compressed(), 1))
    # maxStd = max(np.percentile(fromStd.compressed(), 99), np.percentile(toStd.compressed(), 99))

    # # plot height rasters
    # f, (axFrom, axTo) = plt.subplots(1, 2, figsize=(16,9))   

    # im = axFrom.imshow(fromHeight,
    #                 cmap='jet',
    #                 extent=fromLRBT,
    #                 vmin=minHeight,
    #                 vmax=maxHeight)
    # axFrom.set_xlim(LRBT[0], LRBT[1])
    # axFrom.set_ylim(LRBT[2], LRBT[3])
    # axFrom.set_title('Height: From')

    # im = axTo.imshow(toHeight,
    #                 cmap='jet',
    #                 extent=toLRBT,
    #                 vmin=minHeight,
    #                 vmax=maxHeight)   
    # axTo.set_xlim(LRBT[0], LRBT[1])
    # axTo.set_ylim(LRBT[2], LRBT[3])
    # axTo.set_title('Height: To')

    # f.colorbar(im, 
    #         ax=[axFrom, axTo], 
    #         orientation='horizontal', 
    #         aspect=40)

    # # plot std rasters
    # f, (axFrom, axTo) = plt.subplots(1, 2, figsize=(16,9))    
    
    # im = axFrom.imshow(fromStd,
    #                 cmap='jet',
    #                 extent=fromLRBT,
    #                 vmin=minStd,
    #                 vmax=maxStd)
    # axFrom.set_xlim(LRBT[0], LRBT[1])
    # axFrom.set_ylim(LRBT[2], LRBT[3])
    # axFrom.set_title('Std: From')

    # im = axTo.imshow(toStd,
    #                 cmap='jet',
    #                 extent=toLRBT,
    #                 vmin=minStd,
    #                 vmax=maxStd)   
    # axTo.set_xlim(LRBT[0], LRBT[1])
    # axTo.set_ylim(LRBT[2], LRBT[3])
    # axTo.set_title('Std: To')

    # f.colorbar(im, 
    #     ax=[axFrom, axTo], 
    #     orientation='horizontal', 
    #     aspect=40)

    # plt.show()

    # # close rasterio connections    
    # fromRaster.close()
    # toRaster.close()