import numpy as  np
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
from synthetic_dem_functions import create_dem, export_geotiff


# Set DEM and uncertainty options
deform_params = {
    'tx': 1.7,
    'ty': 1.7,
    'sx': 0,
    'sy': 0.0,
    'g_maj': 30,
    'g_min': 30,
    'g_amp': 0,
    'g_az': 0
}
noise_std = 0.1
X, Y, Z, N = create_dem(250, 5000, 5, 6, deform=deform_params, noise=noise_std)
filename = 'test/transX1p7Y1p7'

# Export DEM and uncertainty array to GeoTIFF
export_geotiff(X, Y, Z, filename + '.tif')
N_uniform = np.ones(Z.shape) * noise_std
export_geotiff(X, Y, N_uniform, filename + 'Std.tif')

# Visualize DEM and uncertainty
image_source = rasterio.open(filename + '.tif')
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

image_source = rasterio.open(filename + 'Std.tif')
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