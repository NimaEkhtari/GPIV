import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
import rasterio.plot
import numpy as np
import json
import os


class create_polygon:

    def __init__(self):
        # some lists to hold screen and geospatial coordinates 
        self.xScalar = list()
        self.yScalar = list()
        self.xData = list()
        self.yData = list()

        # get background raster data
        fromRaster = rasterio.open('from.tif')
        fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
        fromLRBT = list(rasterio.plot.plotting_extent(fromRaster)) # LRBT = [left, right, bottom, top]

        # set up the figure
        self.fig, self.ax = plt.subplots()        
        im = self.ax.imshow(fromHeight,
                        extent=fromLRBT,
                        cmap='jet')
        self.ax.set_xlim(fromLRBT[0], fromLRBT[1])
        self.ax.set_ylim(fromLRBT[2], fromLRBT[3])
        self.ax.set_title('Height: From')
        self.fig.colorbar(im,
                        orientation='vertical', 
                        aspect=40)

        # add a line to the figure with dummy data
        self.line, = self.ax.plot([0][0], color='white', lw=1, alpha=0.3)

        # import existing polygon data or create dummy data
        if os.path.isfile('polygon.json'):
            jsonIn = json.load(open("polygon.json", "r"))
            polyData = np.asarray(jsonIn)
            print('Existing polygon loaded.')
        else:
            polyData = [[0,0],[0,0]]
        
        # add the polygon to the figure
        self.poly = patches.Polygon(polyData, facecolor='0.9', edgecolor='1.0', alpha = 0.3)        
        self.ax.add_artist(self.poly)

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
            # make sure the poly patch is empty if we are tracing a new polygon
            if not self.xScalar:
                self.poly.set_xy([[0,0],[0,0]])

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

        # check for middle or right click (draw polgyon)
        if event.button>1:
            # check for at least three vertices
            if len(self.xData) > 2:

                # insert vertices into polygon patch
                xyData = np.column_stack((self.xData, self.yData))            
                self.poly.set_xy(xyData)

                # clear line data
                self.xScalar = list()
                self.yScalar = list()
                self.xData = list()
                self.yData = list()
                self.line.set_data(self.xData, self.yData)  

                # update figure    
                self.fig.canvas.draw()
                
                # save polygon
                jsonOut = xyData.tolist()
                json.dump(jsonOut, open("polygon.json", "w"))
                print('Polygon vertices saved to file polygon.json')

            else: 
                polyData = self.poly.get_xy()
                if len(polyData) < 4: # a valid polygon does not exist
                    print('Minimum of three vertices are required.')


    def __onKeyPress(self, event):
        # clear line and polygon if escape button pressed
        if event.key == 'escape':
            if self.xScalar:
                # clear existing line data
                self.xScalar = list()
                self.yScalar = list()
                self.xData = list()
                self.yData = list()
                self.line.set_data(self.xData, self.yData)
                
                # update figure
                self.fig.canvas.draw()

                print('Polygon vertices cleared.')
