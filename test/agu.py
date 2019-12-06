import numpy as  np
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
from synthetic_dem_functions import create_dem, export_geotiff


filename = 'test/base.tif'
image_source = rasterio.open(filename)
dem_img = image_source.read(1)
geo_extents = list(rasterio.plot.plotting_extent(image_source)) # Geo extent order is [left, right, bottom, top]
image_source.close()
image_data_min = min(np.percentile(dem_img, 1),
                        np.percentile(dem_img, 1))
image_data_max = max(np.percentile(dem_img, 99),
                        np.percentile(dem_img, 99))

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_proj_type('ortho')
X = np.arange(0, dem_img.shape[0])
Y = np.arange(0, dem_img.shape[1])
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, dem_img, cmap=cm.coolwarm, linewidth=0.5, edgecolors='black')
plt.show()

plt.imshow(dem_img,
            cmap=plt.cm.gray,
            extent=geo_extents,
            vmin=image_data_min,
            vmax=image_data_max)
plt.show()