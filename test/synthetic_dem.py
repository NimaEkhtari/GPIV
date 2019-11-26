import numpy as  np
import matplotlib.pyplot as plt
from osgeo import gdal
import rasterio
import rasterio.plot


def create_dem(dem_size, num_gsn, gsn_max, seed, **kwargs):

    half_data_size = np.floor(dem_size*2/2)

    # Set random seed
    np.random.seed(seed)

    # Uniform random distributions of the gaussian magnitudes and widths
    mag = -gsn_max + 2*gsn_max*np.random.rand(num_gsn, 1)
    wid = mag

    # Uniform random distribution of the gaussian locations
    x_gsn = -half_data_size + 2*half_data_size*np.random.rand(num_gsn, 1)
    y_gsn = -half_data_size + 2*half_data_size*np.random.rand(num_gsn, 1)

    # X, Y coords of synthetic DEM
    half_dem_size = np.floor(dem_size/2)
    x_vec = np.arange(-half_dem_size, half_dem_size+1, 1)
    y_vec = np.arange(-half_dem_size, half_dem_size+1, 1)
    X, Y = np.meshgrid(x_vec, y_vec)

    # Apply deformations to coords
    if 'deform' in kwargs:
        deform_params = kwargs.get('deform')
        dX, dY = get_deformation(X, Y, deform_params)
        X -= dX
        Y -= dY

    # Create DEM surface elevations from random gaussians
    Z = np.zeros(X.shape)
    for i in range(num_gsn):
        Z += mag[i]*np.exp(-((((X-x_gsn[i])**2) / (2*wid[i]**2))
                             +(((Y-y_gsn[i])**2) / (2*wid[i]**2))))

    # Restore X and Y coordinates if deformation was applied
    if 'deform' in kwargs:
        X += dX
        Y += dY

    # Add noise to DEM surface
    if 'noise' in kwargs:
        noise_std = kwargs.get('noise')
        N = noise_std*np.random.randn(Z.shape[0], Z.shape[1])
        Z += N
    else:
        N = []

    return X, Y, Z, N


def get_deformation(X, Y, deform_params):
    """
    Compute deformation at passed X, Y coordinate locations. Deformation
    consists of X and Y direction translations and a deformation whose
    magnitude is Gaussian distributed in space (centered at X=0, Y=0) with a 
    direction set by the passed azimuth (g_az).
    
    ACCEPTS:
    X = x-coordinate for deformation computation
    Y = y-coordinate for deformation computation
    deform_params = dictionary of the following deformation parameters:
        tx = horizontal translation
        ty = vertical translation
        sx = horizontal shear matrix component
        sy = vertical shear matrix component
        g_maj = semi-major axis of 2d spatial Gaussian deformation (horizontal
                at zero degrees azimuth)
        g_min = semi-minor axis of 2d spatial Guassian deformation (vertical
                at zero degrees azimuth)
        g_amp = amplitude of 2d spatial Gaussian deformation
        g_az = azimuth at which the Gaussian deformation is applied

    RETURNS:
    dX = deformation in x direction for each passed coordinate location
    dY = deformation in y direction for each passed coordinate location
    """
    tx = deform_params['tx']
    ty = deform_params['ty']
    sx = deform_params['sx']
    sy = deform_params['sy']
    g_maj = deform_params['g_maj']
    g_min = deform_params['g_min']
    g_amp = deform_params['g_amp']
    g_az = deform_params['g_az']

    dX = np.zeros(X.shape)
    dY = dX.copy()

    # Shear
    shear_matrix = np.array([[0, sx], [sy, 0]])
    XY = np.vstack((X.reshape(1, -1), Y.reshape(1, -1)))
    dXdY = np.matmul(shear_matrix, XY)
    dX += np.reshape(dXdY[0,:], X.shape)
    dY += np.reshape(dXdY[1,:], Y.shape)
    # Translation
    dX += tx
    dY += ty
    # Gaussian deformation
    g_mag = g_amp*np.exp(-(X**2/(2*g_maj**2) + Y**2/(2*g_min**2)))
    dX += g_mag*np.sin(np.radians(g_az))
    dY += g_mag*np.cos(np.radians(g_az))

    return dX, dY


def export_geotiff(X, Y, Z, fname):
    # https://gis.stackexchange.com/questions/37238/writing-numpy-array-to-raster-file
    xmin, ymin, xmax, ymax = [X.min(), Y.min(), X.max(), Y.max()]
    nrows, ncols = np.shape(Z)
    xres = (xmax-xmin) / float(ncols-1)
    yres = (ymax-ymin) / float(nrows-1)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    output_raster = gdal.GetDriverByName('GTiff').Create(fname, ncols, nrows, 1, gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    output_raster.GetRasterBand(1).WriteArray(Z)
    output_raster.FlushCache()



deform_params = {
    'tx': 0,
    'ty': 0,
    'sx': 0,
    'sy': 0,
    'g_maj': 30,
    'g_min': 30,
    'g_amp': 2,
    'g_az': 0
}
noise_std = 0.1

X, Y, Z, N = create_dem(200, 2000, 5, 1, deform=deform_params, noise=noise_std)
export_geotiff(X, Y, Z, 'after.tif')
N_uniform = np.ones(Z.shape) * noise_std
export_geotiff(X, Y, N_uniform, 'afterStd.tif')

image_source = rasterio.open('after.tif')
dem_img = image_source.read(1)
geo_extents = list(rasterio.plot.plotting_extent(image_source)) # Geo extent order is [left, right, bottom, top]
image_source.close()
image_data_min = min(np.percentile(dem_img, 1),
                        np.percentile(dem_img, 1))
image_data_max = max(np.percentile(dem_img, 99),
                        np.percentile(dem_img, 99))
ax1 = plt.subplot(1,2,1)
ax1.imshow(dem_img,
            cmap=plt.cm.gray,
            extent=geo_extents,
            vmin=image_data_min,
            vmax=image_data_max)

image_source = rasterio.open('afterStd.tif')
std_img = image_source.read(1)
image_data_min = min(np.percentile(std_img, 1),
                        np.percentile(std_img, 1))
image_data_max = max(np.percentile(std_img, 99),
                        np.percentile(std_img, 99))
ax2 = plt.subplot(1,2,2)
ax2.imshow(std_img,
            cmap=plt.cm.gray,
            extent=geo_extents,
            vmin=image_data_min,
            vmax=image_data_max)

plt.show()