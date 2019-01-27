import click
import pdal
import json
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from osgeo import gdal

# workaround for click not returning help if no arguments or options passed
# https://stackoverflow.com/questions/50442401/how-to-set-the-default-option-as-h-for-python-click
class DefaultHelp(click.Command):
    def __init__(self, *args, **kwargs):
        context_settings = kwargs.setdefault('context_settings', {})
        if 'help_option_names' not in context_settings:
            context_settings['help_option_names'] = ['-h', '--help']
        self.help_flag = context_settings['help_option_names'][0]
        super(DefaultHelp, self).__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if not args:
            args = [self.help_flag]
        return super(DefaultHelp, self).parse_args(ctx, args)


@click.command(cls=DefaultHelp)
# @click.option('-r', '--raster',
#     type=(click.Path(exists=True, dir_okay=False), click.Path(exists=True, dir_okay=False), float), 
#     help='<from las file> <to las file> <raster size>')
@click.argument('fromLAS', type=click.Path(exists=True, dir_okay=False))
@click.argument('toLAS', type=click.Path(exists=True, dir_okay=False))
@click.argument('rasterSize', type=float)
def raster(fromLAS, toLAS, rasterSize):       

    # r = raster[2]*math.sqrt(0.5)

    # fromRaster = {
    #     "pipeline": [
    #         raster[0],
    #         {
    #             "resolution": raster[2],
    #             "radius": r,
    #             "filename": "from.tif"
    #         }
    #     ]
    # }
    # fromRaster = json.dumps(fromRaster)

    # toRaster = {
    #     "pipeline": [
    #         raster[1],
    #         {
    #             "resolution": raster[2],
    #             "radius": r,
    #             "filename": "to.tif"
    #         }
    #     ]
    # }
    # toRaster = json.dumps(toRaster)
    
    # pipeline = pdal.Pipeline(fromRaster)
    # pipeline.validate()
    # pipeline.execute()

    # pipeline = pdal.Pipeline(toRaster)
    # pipeline.validate()
    # pipeline.execute()
   
    import rasterio
    import rasterio.plot
    import matplotlib.pyplot as plt

    image1 = plt.subplot(121)
    image2 = plt.subplot(122)
    
    fromRaster = rasterio.open('from.tif')
    rasterio.plot.show((fromRaster, 1), ax=image1, title='From')
    toRaster = rasterio.open('to.tif')
    rasterio.plot.show((toRaster, 1), ax=image2, title='To')
    plt.show()


    # image1 = plt.subplot(121)
    # image2 = plt.subplot(122)

    # #read the image files (png files preferred)
    # img_source1 = mpimg.imread('img1.jpg')
    # img_source2 = mpimg.imread('img2.jpg')
    # #put the images into the window
    # _ = image1.imshow(img_source1)
    # _ = image2.imshow(img_source2)

    # #hide axis and show window with images
    # # image1.axis("off")
    # # image2.axis("off")
    # plt.show()


if __name__ == '__main__':
    raster()
