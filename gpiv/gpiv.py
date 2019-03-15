'''
gpiv

Usage:
  gpiv.py raster <fromLAS> <toLAS> <rasterSize>
  gpiv.py polygon 
  gpiv.py piv <templateSize> <stepSize> [--propagate]
  gpiv.py show

Options:
  -h --help       Show this screen.
  -v --version    Show version.
  -p --propagate  Propagate raster height error.
  '''
from docopt import docopt
import raster_option
import polygon_option
import piv_option
import show_option
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
    raster_option.create_rasters(arguments['<fromLAS>'], arguments['<toLAS>'], arguments['<rasterSize>'])

    # display height and error rasters
    # raster_option.show_rasters()

  if arguments['polygon']:
    # will want to eventually allow users to optionally input paths to their own geotiffs

    # check from.tif exists and is geotiff
    
    # check to.tif exists and is geotiff

    # call select
    polygon_option.create_polygon()

  if arguments['piv']:

    # check from.tif exists and is geotiff
    
    # check to.tif exists and is geotiff

    # call piv
    piv_option.piv(float(arguments['<templateSize>']), float(arguments['<stepSize>']))

    # print(arguments)
  
  if arguments['show']:

    show_option.show_rasters()
