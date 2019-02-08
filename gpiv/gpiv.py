'''
gpiv

Usage:
  gpiv.py raster <fromLAS> <toLAS> <rasterSize>
  gpiv.py select 
  gpiv.py piv <windowSize> <stepSize> [--propagate]
  gpiv.py show

Options:
  -h --help       Show this screen.
  -v --version    Show version.
  -p --propagate  Propagate raster height error.
  '''
from docopt import docopt
import raster_option
import select_option6
import os
import sys
import rasterio
import numpy as np

def is_number(n):
  try:
    x = float(n)
    if x <= 0:
      return False
  except ValueError:
      return False
  return True

if __name__ == '__main__':

  arguments = docopt(__doc__)

  if arguments['raster']: 

    # check fromFile is valid
    if not os.path.isfile(arguments['<fromLAS>']):
      print('Invalid fromLAS file.')
      sys.exit()

    # check toFile is valid
    if not os.path.isfile(arguments['<toLAS>']):
      print('Invalid toLAS file.')
      sys.exit()

    # check rasterSize is >0
    if not is_number(arguments['<rasterSize>']):
      print('Raster size must be a positive number')
      sys.exit()
  
    # raster LAS files
    # raster_option.create_rasters(arguments['<fromLAS>'], arguments['<toLAS>'], arguments['<rasterSize>'])

    # display height and error rasters
    raster_option.show_rasters()

  if arguments['select']:
    # will want to eventually allow users to optionally input their own geotiffs

    # check from.tif exists and is geotiff
    
    # check to.tif exists and is geotiff

    # call select
    # fromRaster = rasterio.open('from.tif')
    # fromHeight =  fromRaster.read(3, masked=True) # read band to numpy array
    # fromHeight = (fromHeight - fromHeight.min()) / (fromHeight.max() - fromHeight.min()) # normalize to [0,1]
    # fromHeight = fromHeight * 255
    # fromImg = fromHeight.astype(np.uint8)
    # fromImg = np.stack((fromImg,)*3, axis=-1)

    # toRaster = rasterio.open('to.tif')
    # toHeight =  toRaster.read(3, masked=True) # read band to numpy array
    # toHeight = (toHeight - toHeight.min()) / (toHeight.max() - toHeight.min()) # normalize to [0,1]
    # toHeight = toHeight * 255
    # toImg = toHeight.astype(np.uint8)
    # toImg = np.stack((toImg,)*3, axis=-1)

    # area = select_option4.polygon_drawer('From', 'To')
    # area.run(fromImg, toImg)

    # print(arguments)

    pivArea = select_option6.create_polygon()
    pivArea.run()

  if arguments['piv']:

    # check from.tif exists and is geotiff
    
    # check to.tif exists and is geotiff

    # call piv
    # piv()

    print(arguments)
