
import piv_functions
import show_functions



def piv(before_height, after_height, template_size, step_size, prop, outname):
    '''
    Runs PIV on a pair pre- and post-event DEMs.

    \b
    Arguments: BEFORE_HEIGHT  Pre-event DEM in GeoTIFF format
               AFTER_HEIGHT   Post-event DEM in GeoTIFF format
               TEMPLATE_SIZE  Size of square correlation template in pixels
               STEP_SIZE      Size of template step in pixels
    '''
    if prop:
        propagate = True
        before_uncertainty = prop[0]
        after_uncertainty = prop[1]
    else:
        propagate = False
        before_uncertainty = ''
        after_uncertainty = ''
    
    if outname:
        output_base_name = outname + '_'
    else:
        output_base_name = ''

    piv_functions.piv(before_height, after_height, 
                      template_size, step_size, 
                      before_uncertainty, after_uncertainty,
                      propagate, output_base_name)



before_height = r'.\example_data\height_2001.tif'
after_height  = r'.\example_data\height_2015.tif'
before_uncertainty = r'.\example_data\uncertainty_2001.tif'
after_uncertainty  = r'.\example_data\uncertainty_2015.tif'

template_size = 50
step_size = 50
prop = True

piv_functions.piv(before_height, after_height, 
                  template_size, step_size, 
                  before_uncertainty, after_uncertainty,
                  prop, 'test_')