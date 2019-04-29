'''
gpiv

Usage:
	gpiv.py raster <lasFile> <rasterSize> (--from | --to)
	gpiv.py piv <templateSize> <stepSize> [--propagate]
	gpiv.py show (--from | --to) (--height | --error) [(--vectors <vectorScaleFactor>)] [(--ellipses <ellipseScaleFactor>)]

Options:
	--help          Show this screen.
	--propagate     Propagate raster error.
	--from          'From' data.
	--to            'To' data.
	--height        Height data.
	--error         Error data.
	--vectors       Show PIV displacement vectors. You must supply a scale factor.
	--ellipses      Show propagated PIV displacement uncertainty ellipses. You must supply a scale factor.
'''

from docopt import docopt
from raster_option import create_rasters
# import .polygon_option
from piv_option import  piv
from show_option import show
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
		# check lasFile is valid
		if not os.path.isfile(arguments['<lasFile>']):
			print('Invalid LAS file.')
			sys.exit()

		# check rasterSize is positive number
		if not is_positive_number(arguments['<rasterSize>']):
			print('Raster size must be a positive number')
			sys.exit()
	
		# raster LAS files
		create_rasters(arguments['<lasFile>'], arguments['<rasterSize>'], arguments['--from'], arguments['--to'])

		# display height and error rasters
		# raster_option.show_rasters()

	# if arguments['polygon']:
	# 	# will want to eventually allow users to optionally input paths to their own geotiffs

	# 	# check from.tif exists and is geotiff
		
	# 	# check to.tif exists and is geotiff

	# 	# call select
	# 	polygon_option.create_polygon()

	if arguments['piv']:
		# check from.tif exists and is geotiff
		
		# check to.tif exists and is geotiff

		# check that templateSize and stepSize are positive integers

		# run piv
		piv(int(arguments['<templateSize>']), int(arguments['<stepSize>']), arguments['--propagate'])
	
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
		
		show(arguments['--from'], arguments['--to'],
		     arguments['--height'], arguments['--error'],
			 arguments['--vectors'], arguments['<vectorScaleFactor>'],
			 arguments['--ellipses'], arguments['<ellipseScaleFactor>'])
