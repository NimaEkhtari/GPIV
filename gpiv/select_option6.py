import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
import numpy as np


class create_polygon:

    def __init__(self):
        self.xScalar = list()
        self.yScalar = list()
        self.xData = list()
        self.yData = list()

    def __onButtonClick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        self.xScalar.append(event.x)
        self.yScalar.append(event.y)
        self.xData.append(event.xdata)
        self.yData.append(event.ydata)

    def __onKeyPress(self, event):
        print('you pressed', event.key)

    def run(self):
    
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
        fig, (axFrom, axTo) = plt.subplots(1, 2, figsize=(16,9))   

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

        fig.colorbar(im, 
                ax=[axFrom, axTo], 
                orientation='horizontal', 
                aspect=40)

        cid1 = fig.canvas.mpl_connect('button_press_event', self.__onButtonClick)
        cid2 = fig.canvas.mpl_connect('key_press_event', self.__onKeyPress)

        plt.show()


