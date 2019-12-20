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
fig = plt.figure(figsize=(4.75, 2))
ax = fig.gca(projection='3d')
ax.set_proj_type('ortho')
ax.view_init(elev=70, azim=-130)
X = np.arange(0, dem_img.shape[0])
Y = np.arange(0, dem_img.shape[1])
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, dem_img, cmap=cm.coolwarm, linewidth=0.3, edgecolors='black', rstride=8, cstride=8)
ax.set_zticks([0, 10])
ax.set_zticklabels(['0', '10'], fontfamily='serif', fontsize=8)
plt.xticks([0,100,200], ['0', '100', '200'], fontfamily='serif', fontsize=8)
plt.yticks([0,100,200], ['0', '100', '200'], fontfamily='serif', fontsize=8)
ax.tick_params(axis='both', pad=-2)
# plt.xlabel('X (pixels)', fontfamily='serif', labelpad=12, fontsize=8)
plt.savefig('test/base.png', dpi=600)
plt.show()


# plt.imshow(dem_img,
#             cmap=plt.cm.gray,
#             extent=geo_extents,
#             vmin=image_data_min,
#             vmax=image_data_max)
# plt.show()



# # Pixel-Locking
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# ty = np.arange(0, 2.1, 0.1)
# sx = [0.038, 0.035, 0.038, 0.037, 0.036, 0.036, 0.037, 0.037, 0.038, 0.038, 0.038, 0.038, 0.037, 0.037, 0.036, 0.036, 0.037, 0.037, 0.038, 0.038, 0.038]
# sy = [0.040, 0.039, 0.036, 0.033, 0.028, 0.026, 0.029, 0.034, 0.037, 0.040, 0.040, 0.039, 0.037, 0.034, 0.029, 0.027, 0.030, 0.034, 0.038, 0.039, 0.040]
# bx = [66, 68, 68, 69, 71, 68, 67, 66, 65, 64, 64, 65, 66, 66, 69, 69, 67, 65, 65, 65, 65]
# by = [70, 70, 72, 78, 84, 88, 82, 77, 74, 70, 69, 70, 74, 77, 83, 86, 82, 76, 74, 69, 68]
# bm = [69, 70, 73, 76, 80, 83, 82, 76, 74, 72, 71, 71, 74, 76, 79, 83, 80, 76, 73, 71, 71]
# # fig, ax1 = plt.subplots(figsize=(4, 2))
# # ax1.set_xlabel('Y Disp. (px)')
# # ax1.set_ylabel('RMSE Y Disp. (px)', color='blue')
# # ax1.plot(ty, sy, 'o-', color='blue')
# # ax1.tick_params(axis='y', labelcolor='blue')
# # ax1.set_ylim(0.02, 0.05)
# # ax2 = ax1.twinx()
# # ax2.set_ylabel('Y Bounded (%)', color='red')
# # ax2.plot(ty, by, 'o-', color='red')
# # ax2.tick_params(axis='y', labelcolor='red')
# # fig.tight_layout()
# # plt.show()
# fig, ax1 = plt.subplots(figsize=(4, 2))
# ax1.set_xlabel('Y Disp. (px)')
# ax1.set_ylabel('Y Bounded (%)')
# ax1.plot(ty, by, 'o-', color='blue')
# ax1.set_ylim(60, 100)
# ax1.set_title('Peak-Locking Influence')
# # ax1.grid()
# fig.tight_layout()
# plt.show()