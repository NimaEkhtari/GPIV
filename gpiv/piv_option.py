import rasterio
import rasterio.plot

def get_extents(f, t):
    # determine teh Left, Right, Bottom,and Top extents of the overlapping DEM data
    fLRBT = list(rasterio.plot.plotting_extent(f)) # LRBT = [left, right, bottom, top]
    tLRBT = list(rasterio.plot.plotting_extent(t))

    extentsLRBT = list()
    extentsLRBT.append(max(fLRBT[0], tLRBT[0]))
    extentsLRBT.append(min(fLRBT[1], tLRBT[1]))
    extentsLRBT.append(max(fLRBT[2], tLRBT[2]))
    extentsLRBT.append(min(fLRBT[3], tLRBT[3]))
    
    return extentsLRBT

def piv():
    # get the data
    fromRaster = rasterio.open('from.tif')
    fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
    fromStd = fromRaster.read(6, masked=True)

    toRaster = rasterio.open('to.tif')
    toHeight = toRaster.read(3, masked=True)
    toStd = toRaster.read(6, masked=True) 

    # determine overlap extents
    extentsLRBT = get_extents(fromRaster, toRaster)

    # determine number of template window movements in horizontal (u) and vertical (v)

