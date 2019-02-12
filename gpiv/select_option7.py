import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import rasterio
import rasterio.plot
import numpy as np
import json


class create_polygon:

    def __init__(self):
        # some lists to hold screen and geospatial coordinates 
        self.xScalar = list()
        self.yScalar = list()
        self.xData = list()
        self.yData = list()

        # get the data
        fromRaster = rasterio.open('from.tif')
        fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array

        # form the figure
        self.fig, self.ax = plt.subplots()        
        im = self.ax.imshow(fromHeight,
                        cmap='jet')
                        # vmin=min(np.percentile(fromHeight.compressed(), 1)),
                        # vmax=min(np.percentile(fromHeight.compressed(), 99)))
        self.ax.set_title('Height: From')
        self.fig.colorbar(im,
                        orientation='vertical', 
                        aspect=40)

        # get the line ready
        self.line, = self.ax.plot([0][0], color='white', lw=1, alpha=0.3)

        # define mouse click and key press functions
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.__onButtonClick)
        self.cid2 = self.fig.canvas.mpl_connect('key_press_event', self.__onKeyPress)

        plt.show()
      

    def __onButtonClick(self, event):        
        # ignore mouse clicks outside the axes
        if event.inaxes is None: 
            return
        
        # check for left click (add line segment)
        if event.button==1:
            # append screen and spatial coordinates
            self.xScalar.append(event.x)
            self.yScalar.append(event.y)
            self.xData.append(event.xdata)
            self.yData.append(event.ydata)

            # draw line
            self.line.set_data(self.xData, self.yData)
            self.line.figure.canvas.draw()
            
            print('Point appended to polygon: x=%d, y=%d, xdata=%f, ydata=%f' %
                (event.x, event.y, event.xdata, event.ydata))

        # check for middle or right click(draw pologyon)
        if event.button>1:
            # remove line
            self.line.remove()

            # draw polygon
            xyData = np.column_stack((self.xData, self.yData))            
            poly = Polygon(xyData, 
                            facecolor='0.9', 
                            edgecolor='1.0',
                            alpha = 0.3)
            self.ax.add_patch(poly)         
            self.fig.canvas.draw()
            
            # save polygon
            jsonOut = xyData.tolist()
            json.dump(jsonOut, open("polygon.json", "w"))
            print('polygon vertices saved to file polygon.json')

            # exit
            self.fig.canvas.mpl_disconnect(self.cid1)


    def __onKeyPress(self, event):
        # clear line and polygon if escape button pressed
        if event.key == 'escape':
            if self.xScalar:
                self.xScalar = list()
                self.yScalar = list()
                self.xData = list()
                self.yData = list()

                self.line.remove()
                self.fig.canvas.draw()

                self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.__onButtonClick)



    # def run(self):
    
        # # get the data
        # fromRaster = rasterio.open('from.tif')
        # fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
        # fromStd = fromRaster.read(6, masked=True)

        # toRaster = rasterio.open('to.tif')
        # toHeight = toRaster.read(3, masked=True)
        # toStd = toRaster.read(6, masked=True) 
            
        # # Determine common bounds for 'to' and 'from' raster spatial extents and
        # # values so that differences in extents and values are apparent when plotted.
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
        # self.fig, (self.axFrom, self.axTo) = plt.subplots(1, 2, figsize=(16,9))   

        # im = self.axFrom.imshow(fromHeight,
        #                 cmap='jet',
        #                 extent=fromLRBT,
        #                 vmin=minHeight,
        #                 vmax=maxHeight)
        # self.axFrom.set_xlim(LRBT[0], LRBT[1])
        # self.axFrom.set_ylim(LRBT[2], LRBT[3])
        # self.axFrom.set_title('Height: From')

        # im = self.axTo.imshow(toHeight,
        #                 cmap='jet',
        #                 extent=toLRBT,
        #                 vmin=minHeight,
        #                 vmax=maxHeight)   
        # self.axTo.set_xlim(LRBT[0], LRBT[1])
        # self.axTo.set_ylim(LRBT[2], LRBT[3])
        # self.axTo.set_title('Height: To')

        # self.fig.colorbar(im, 
        #         ax=[self.axFrom, self.axTo], 
        #         orientation='horizontal', 
        #         aspect=40)

        # self.lineFrom, = self.axFrom.plot([0][0], color='white', lw=1)
        # self.lineTo, = self.axTo.plot([0][0], color='white', lw=1)

        # self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.__onButtonClick)
        # cid2 = self.fig.canvas.mpl_connect('key_press_event', self.__onKeyPress)

        # plt.show()


