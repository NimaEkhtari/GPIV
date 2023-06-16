
import piv_functions
import show_functions
import pdal
import json
import numpy as np
import matplotlib.pyplot as plt
from tin_interpolation import estimate_uncertainty
from rasters import write_raster


# def piv(before_height, after_height, template_size, step_size, prop, outname):
#     '''
#     Runs PIV on a pair pre- and post-event DEMs.

#     \b
#     Arguments: BEFORE_HEIGHT  Pre-event DEM in GeoTIFF format
#                AFTER_HEIGHT   Post-event DEM in GeoTIFF format
#                TEMPLATE_SIZE  Size of square correlation template in pixels
#                STEP_SIZE      Size of template step in pixels
#     '''
#     if prop:
#         propagate = True
#         before_uncertainty = prop[0]
#         after_uncertainty = prop[1]
#     else:
#         propagate = False
#         before_uncertainty = ''
#         after_uncertainty = ''
    
#     if outname:
#         output_base_name = outname + '_'
#     else:
#         output_base_name = ''

#     piv_functions.piv(before_height, after_height, 
#                       template_size, step_size, 
#                       before_uncertainty, after_uncertainty,
#                       propagate, output_base_name)



# def get_points(las):
#     p = pdal.Reader(las, use_eb_vlr=True).pipeline()
#     p.execute()
#     las = p.get_dataframe(0)

#     points = np.transpose(np.vstack([las['X'], las['Y'], las['Z']]))
#     cov = np.transpose(np.vstack([las['VarianceX'], las['VarianceY'],
#                                   las['VarianceZ'], las['CovarianceXY'],
#                                   las['CovarianceXZ'], las['CovarianceYZ']]))

#     minx = p.metadata['metadata']['readers.las']['minx']
#     maxx = p.metadata['metadata']['readers.las']['maxx']
#     miny = p.metadata['metadata']['readers.las']['miny']
#     maxy = p.metadata['metadata']['readers.las']['maxy']
#     bounds = [minx, maxx, miny, maxy]
#     return (points, cov, bounds)


# # before_height = r'.\example_data\height_2001.tif'
# # after_height  = r'.\example_data\height_2015.tif'
# # before_uncertainty = r'.\example_data\uncertainty_2001.tif'
# # after_uncertainty  = r'.\example_data\uncertainty_2015.tif'
# # template_size = 50
# # step_size = 50
# # prop = True






# ''' ----------------------------------------------------------------------- '''
# '''                           Input variables                               '''
# ''' ----------------------------------------------------------------------- '''

# cell_size = 2
# margin = 5          # margin around the dataset to avoid empty TIN triangles (in pixels)
# epsg = 6344 #32615
# before_point_cloud = r'./data/input/fixed.laz'
# after_point_cloud = r'./data/input/moving.laz'
before_dem = r'./data/output/before_dem.tif'
before_tpu = r'./data/output/before_tpu.tif'
after_dem = r'./data/output/after_dem.tif'
after_tpu = r'./data/output/after_tpu.tif'


# p_before, tpu_before, bounds_before = get_points(before_point_cloud)
# [minxb, maxxb, minyb, maxyb] = bounds_before[0 : 4]

# p_after, tpu_after, bounds_after = get_points(after_point_cloud)
# [minxa, maxxa, minya, maxya] = bounds_after[0 : 4]

# minx, maxx = max(minxa, minxb), min(maxxa, maxxb)
# miny, maxy = max(minya, minyb), min(maxya, maxyb)
# del minxa, minxb, maxxa, maxxb, minya, minyb, maxya, maxyb


# offset = margin * cell_size
# mingrdX, grddX, maxgrdX = np.ceil(minx + offset), cell_size, np.floor(maxx - offset)
# mingrdY, grddY, maxgrdY = np.ceil(miny + offset), cell_size, np.floor(maxy - offset)


# ''' ----------------------------------------------------------------------- '''
# grdX = np.arange(mingrdX + grddX, maxgrdX + grddX, grddX) - grddX / 2
# grdY = np.arange(mingrdY + grddY, maxgrdY + grddY, grddY) - grddY / 2

# meshX, meshY = np.meshgrid(grdX, grdY)
# grdpt = np.array(np.column_stack((meshX.ravel(), meshY.ravel())))
# del grddX, maxgrdX, grddY, grdX, grdY



# # Z, dZ = estimate_uncertainty(ptcloud, sig_COV, grdpt)


# # ZZ = Z.reshape(meshX.shape)
# # dem = np.flipud(ZZ)             # Need to flip the raster in height direction to show and save
# # fig = plt.figure()
# # plt.imshow(dem)

# # DZ = dZ.reshape(meshX.shape)
# # tpu = np.flipud(np.sqrt(DZ))    # Need to flip the raster in height direction to show and save. square root caluclates std from variance
# # fig = plt.figure()
# # plt.imshow(tpu, vmin=np.nanmin(tpu), vmax=0.6)



# ''' ----------------------------------------------------------------------- '''
# '''       Writing the interpolated DEM and its TPU to geo-tiff format       '''

# ul = (int(np.min(meshX) - (cell_size / 2)), int(np.max(meshY) + (cell_size / 2)))         # Coordinates of upper left pixel
# pixelWidth = cell_size          # ground pixel size in east-west direction
# pixelHeight = cell_size         # ground pixel size in north-south direction



# Z, dZ = estimate_uncertainty(p_before, tpu_before, grdpt)
# dem_b = np.flipud(Z.reshape(meshX.shape))
# tpu_b = np.flipud(np.sqrt(dZ).reshape(meshX.shape))

# write_raster(before_dem, dem_b, ul, pixelWidth, pixelHeight, epsg)
# write_raster(before_tpu, tpu_b, ul, pixelWidth, pixelHeight, epsg)



# Z, dZ = estimate_uncertainty(p_after, tpu_after, grdpt)
# dem_a = np.flipud(Z.reshape(meshX.shape))
# tpu_a = np.flipud(np.sqrt(dZ).reshape(meshX.shape))

# write_raster(after_dem, dem_a, ul, pixelWidth, pixelHeight, epsg)
# write_raster(after_tpu, tpu_a, ul, pixelWidth, pixelHeight, epsg)

# ''' ----------------------------------------------------------------------- '''
# ''' ----------------------------------------------------------------------- '''
# ''' ----------------------------------------------------------------------- '''
# ''' ----------------------------------------------------------------------- '''
# ''' ----------------------------------------------------------------------- '''
# ''' ----------------------------------------------------------------------- '''
# ''' ----------------------------------------------------------------------- '''
# ''' ----------------------------------------------------------------------- '''



# # before_height = r'.\example_data\height_2001.tif'
# # after_height  = r'.\example_data\height_2015.tif'
# # before_uncertainty = r'.\example_data\uncertainty_2001.tif'
# # after_uncertainty  = r'.\example_data\uncertainty_2015.tif'


template_size = 50
step_size = 25
search_scale = 2      # Additional size of search area compared to template area (1.2 means 20* extra)
prop = True





piv_functions.piv(before_dem, after_dem, 
                  template_size, step_size, search_scale,
                  before_tpu, after_tpu,
                  prop, 'test_')

with open('test_vectors.json') as json_file:
    origins_vectors = json.load(json_file)
ov = np.asarray(origins_vectors)

with open('test_covariances.json') as json_file:
    c = json.load(json_file)
    
