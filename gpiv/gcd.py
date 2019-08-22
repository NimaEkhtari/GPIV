import click
from piv import Piv
import piv_show


@click.group()
def cli():
    pass


@click.command()
@click.argument('before_height')
@click.argument('after_height')
@click.argument('template_size', type=int)
@click.argument('step_size', type=int)
@click.option('-p', nargs=2, help='Option to propagate error. Requires two arguments: 1) pre-event uncertainties in GeoTIFF format, 2) post-event uncertainties in GeoTIFF format.')
@click.option('-o', nargs=1, help='Optional base filename to use for output files.')
def piv(before_height, after_height, template_size, step_size, p, o):
    '''
    Runs PIV on a pair pre- and post-event DEMs.

    \b
    Arguments: BEFORE_HEIGHT  Pre-event DEM in GeoTIFF format
               AFTER_HEIGHT   Post-event DEM in GeoTIFF format
               TEMPLATE_SIZE  Size of square correlation template in pixels
               STEP_SIZE      Size of template step in pixels
    '''
    my_piv = Piv(before_height, after_height, template_size, step_size)
    if p:
        my_piv.propagate = True
        my_piv.before_uncertainty_file = p[0]
        my_piv.after_uncertainty_file = p[1]
    if o:
        my_piv.output_base_name = o + '_'
    my_piv.run()


@click.command()
@click.argument('background_image')
@click.option('-v', nargs=1, help='Option to show PIV vectors. Requires a json file of PIV vectors')
@click.option('-e', nargs=1, help='Option to show PIV uncertainty ellipses. Requires a json file of PIV covariance matrices.')
# may want to add user ability to set explicit scale factor for vectors and/or ellipses (i.e., completely bypass the auto generated scale factor)
@click.option('-s', nargs=1, type=float, help='Option to scale the displayed PIV vectors and/or uncertainty ellipses. Requires a numeric scale factor.')
def pivshow(background_image, v, e, s):
    '''
    Optionally displays PIV displacement vectors and/or uncertainty ellipses over a background image.
    
    Arguments: BACKGROUND_IMAGE  Background image in GeoTIFF format
    '''
    piv_show.show(background_image, v, e, s)


cli.add_command(piv)
cli.add_command(pivshow)

if __name__ == '__main__':
    cli()
