'''
gpiv

Usage:
  gpiv.py raster <fromLAS> <toLAS> <rasterSize>
  gpiv.py polygon 
  gpiv.py piv <templateSize> <stepSize> [--propagate]
  gpiv.py show (--from | --to) (--height | --error) [(--vectors <vectorScaleFactor>)] [(--ellipses <ellipseScaleFactor>)]

Options:
  -h --help       Show this screen.
  -v --version    Show version.
  --propagate     Propagate raster error.
  --from          Show 'from' raster.
  --to            Show 'to' raster.
  --height        Show height raster.
  --error         Show error raster.
  --vectors       Show PIV displacement vectors. You must supply a scale factor.
  --ellipses      Show propagated PIV displacement uncertainty ellipses. You must supply a scale factor.
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

def is_positive_number(n):
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
		if not is_positive_number(arguments['<rasterSize>']):
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

		# check that templateSize and stepSize are integers

		# call piv
		myPiv = piv_option.Piv()
		myPiv.compute(int(arguments['<templateSize>']), int(arguments['<stepSize>']), arguments['--propagate'])
		if arguments['--propagate']:
			myPiv.propagate()

		# print(arguments)
	
	if arguments['show']:
		# check for required file existence
		if arguments['--from'] and arguments['--height'] and not os.path.isfile('fromHeight.tif'):
			print("Missing 'fromHeight.tif' file.")
			sys.exit()

		if arguments['--from'] and arguments['--error'] and not os.path.isfile('fromError.tif'):
			print("Missing 'fromError.tif' file.")
			sys.exit()

		if arguments['--to'] and arguments['--height'] and not os.path.isfile('toHeight.tif'):
			print("Missing 'toHeight.tif' file.")
			sys.exit()

		if arguments['--to'] and arguments['--error'] and not os.path.isfile('toError.tif'):
			print("Missing 'toError.tif' file.")
			sys.exit()
		
		if arguments['--vectors'] and not os.path.isfile('piv_origins_offsets.json'):
			print("PIV vector file 'piv_origins_offset.json' missing.")
			sys.exit()

		if arguments['--ellipses'] and not os.path.isfile('piv_covariance_matrices.json'):
			print("PIV vector file 'piv_covariance matrices.json' missing.")
			sys.exit()

		if arguments['--vectors'] and not is_positive_number(arguments['<vectorScaleFactor>']):
			print('PIV displacement vector scale factor must be greater than 0.')
			sys.exit()

		if arguments['--ellipses'] and not is_positive_number(arguments['<ellipseScaleFactor>']):
			print('PIV error ellipse scale factor must be greater than 0.')
			sys.exit()
		
		show_option.show(arguments['--from'], arguments['--to'],
		             	 arguments['--height'], arguments['--error'],
						 arguments['--vectors'], arguments['<vectorScaleFactor>'],
						 arguments['--ellipses'], arguments['<ellipseScaleFactor>'])
