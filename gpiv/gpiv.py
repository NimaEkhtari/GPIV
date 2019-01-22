import click
import pdal

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
@click.option('-r', '--raster', 
    nargs=2, 
    type=click.Path(exists=True, dir_okay=False), 
    help='<from las file> <to las file>')
def rasterize(raster):    
    click.echo(raster[0])
    
    import json

    x = {
        "pipeline": [
            raster[0],
            {
                "resolution": 5,
                "radius": 7,
                "filename": "outputfile.tif"
            }
        ]
    }

    print(x)
    print(type(x))
    y = json.dumps(x)
    print(y)
    print(type(y))


if __name__ == '__main__':
    rasterize()
