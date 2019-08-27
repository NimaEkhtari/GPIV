import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
import numpy as np

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

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

    line = axFrom.plot([0][0])
    linebuilder = LineBuilder(line)

    # cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

