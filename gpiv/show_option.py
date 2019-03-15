import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
import numpy as np


def show_rasters():

    # get the data
    fromRaster = rasterio.open('from.tif')
    fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
    fromStd = fromRaster.read(6, masked=True)

    toRaster = rasterio.open('to.tif')
    toHeight = toRaster.read(3, masked=True)
    toStd = toRaster.read(6, masked=True) 
        
    # Determine common bounds for 'to' and 'from' raster spatial extents and
    # values so that differences in extents and values are apparent when
    # plotted.
    # 1. Determine bounding box encompassing both the 'from' and 'to' rasters
    fromLRBT = list(rasterio.plot.plotting_extent(fromRaster)) # LRBT = [left, right, bottom, top]
    toLRBT = list(rasterio.plot.plotting_extent(toRaster))
    LRBT = list()
    LRBT.append(min(fromLRBT[0], toLRBT[0]))
    LRBT.append(max(fromLRBT[1], toLRBT[1]))
    LRBT.append(min(fromLRBT[2], toLRBT[2]))
    LRBT.append(max(fromLRBT[3], toLRBT[3]))
    # 2. Get maximum and minimum array values for both the 'from' and 'to' rasters. Using percentiles to avoid outliers.
    minHeight = min(np.percentile(fromHeight.compressed(), 1), np.percentile(toHeight.compressed(), 1))
    maxHeight = max(np.percentile(fromHeight.compressed(), 99), np.percentile(toHeight.compressed(), 99))
    minStd = min(np.percentile(fromStd.compressed(), 1), np.percentile(toStd.compressed(), 1))
    maxStd = max(np.percentile(fromStd.compressed(), 99), np.percentile(toStd.compressed(), 99))

    # plot height rasters
    f, (axFrom, axTo) = plt.subplots(1, 2, figsize=(16,9))   

    im = axFrom.imshow(fromHeight,
                    cmap='jet',
                    extent=fromLRBT,
                    vmin=minHeight,
                    vmax=maxHeight)
    axFrom.set_xlim(LRBT[0], LRBT[1])
    axFrom.set_ylim(LRBT[2], LRBT[3])
    axFrom.set_title('Height: From')

    im = axTo.imshow(toHeight,
                    cmap='jet',
                    extent=toLRBT,
                    vmin=minHeight,
                    vmax=maxHeight)   
    axTo.set_xlim(LRBT[0], LRBT[1])
    axTo.set_ylim(LRBT[2], LRBT[3])
    axTo.set_title('Height: To')

    f.colorbar(im, 
            ax=[axFrom, axTo], 
            orientation='horizontal', 
            aspect=40)

    # plot std rasters
    f, (axFrom, axTo) = plt.subplots(1, 2, figsize=(16,9))    
    
    im = axFrom.imshow(fromStd,
                    cmap='jet',
                    extent=fromLRBT,
                    vmin=minStd,
                    vmax=maxStd)
    axFrom.set_xlim(LRBT[0], LRBT[1])
    axFrom.set_ylim(LRBT[2], LRBT[3])
    axFrom.set_title('Std: From')

    im = axTo.imshow(toStd,
                    cmap='jet',
                    extent=toLRBT,
                    vmin=minStd,
                    vmax=maxStd)   
    axTo.set_xlim(LRBT[0], LRBT[1])
    axTo.set_ylim(LRBT[2], LRBT[3])
    axTo.set_title('Std: To')

    f.colorbar(im, 
        ax=[axFrom, axTo], 
        orientation='horizontal', 
        aspect=40)

    plt.show()

    # close rasterio connections    
    fromRaster.close()
    toRaster.close()