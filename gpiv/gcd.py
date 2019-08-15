# '''
# gcd

# Usage:
# 	gcd.py piv <before_height> <after_height> <template_size> <step_size> [-p=(<before_uncertainty> <after_uncertainty>)] [-o <output_base_name>]
# 	gcd.py pivshow <underlying_image> <displacement_file> <scale_factor> [-e <uncertainty_file> <scale_factor>]
	
# Options:
# 	-h	show this screen.
# 	-p	propagate raster uncertainty
# 	-o	base name for output files
# 	-e	show propagated uncertainty ellipses
# '''

import click
from piv_option2 import Piv


@click.group()
def cli():
	pass

@click.command()
def piv():
	click.echo('piv works')

@click.command()
def pivshow():
	click.echo('pivshow works')

cli.add_command(piv)
cli.add_command(pivshow)

if __name__ == '__main__':
	cli()


	# args = docopt(__doc__)

	# if args['piv']:
		
	# 	my_piv = Piv(args['<before_height>'], args['<after_height>'], args['<template_size>'], args['<step_size>'], args['-p'])
	# 	if my_piv.propagate:
	# 		my_piv.before_uncertainty = args['before_uncertainty']
	# 		my_piv.after_uncertainty = args['after_uncertainty']
		
	# 	# my_piv.run_piv()
	
	# 	print("piv works")
	
	# if args['pivshow']:		

	# 	# show(arguments['--from'], arguments['--to'],
	# 	#      arguments['--height'], arguments['--error'],
	# 	# 	 arguments['--vectors'], arguments['<vectorScaleFactor>'],
	# 	# 	 arguments['--ellipses'], arguments['<ellipseScaleFactor>'])

	# 	print("pivshow works")