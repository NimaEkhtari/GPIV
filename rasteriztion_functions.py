# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:07:46 2023

Developed at NCALM, University of Houston.

This code is developed based on original effort by Luyen Bui:
    https://orcid.org/0000-0003-1091-5573

Based on the following paper:
    https://doi.org/10.1109/LGRS.2021.3134587
    
Partial Derivatives esplained in the supplementary material of above paper at:
    https://doi.org/10.1109/LGRS.2021.3134587/mm1


Note:
    1- Always, the triangle containing the center of the grid cell is used for
       interpolation, tehrefore ff a grid sizes are large and a grid cell 
       contains more than one triangle, there is no confusion as to how the 
       interpolated height is calculated.
    2- It is considered best practice to select the grid cell based on the 
       average point density of the point cloud to avoid over- or under-sampling
       artifacts in the DEM.

@author: nekhtari@uh.edu
"""

import numpy as np
import multiprocessing as mp
import sys
from scipy.spatial import Delaunay
import time
import pdal
import matplotlib.pyplot as plt
from rasters import write_raster


#from temp_funcs import TIN_lin_Err_Prop_many_MP_starmap2Async


def TIN_lin_Err_Prop_many_MP_starmap2Async(points, points_tpu, grids, nprocs=mp.cpu_count()):
    
    inds, tvc = get_triangles(points, grids)
    
    centroid = np.mean(points, axis = 0)
    tv = tvc - centroid
    gr = grids - centroid[0:2]
    
    tin_coeffs = get_tin_coeffs(tv)                                     # calculate TIN interpolation coefficients
    parderiv = get_partial_derivatives(tv, gr, tin_coeffs)              # calculate partial derivatives of TIN interpolation eq w.r.t coordinates of 3 vertices
    
    interpolated_grid = Interpolate_TIN(gr, tin_coeffs, centroid)       # Calculate interpolated heights
    tpu_grid = propagate_tin_error(points_tpu, inds, parderiv)          # Calculate propagated Uncertainty for each grid cell
    return (interpolated_grid, tpu_grid)


def get_triangles(points, grids):
    '''
    function to returns TIN triangle vertex indices and coordinates for each grid cell
    input: 
        points: (n x 3) array of point cloud cooordinates
        grids:  (m x 2) array of horizontal coordinates of grid cell centers
    output:
        inds:   (m x 3) arrat of TIN triangle vertices containing each grid cell
        tvc:    (m x 3 x 3) array of TIN triangle vertex coordinates containing each grid cell
        
    '''
    
    tin = Delaunay(ptcloud[:, 0:2])             # Delaunay TIN
    point_indices = tin.simplices               # Point cloud indices of TIN triangle vertices
    tin_cord = ptcloud[tin.simplices]           # vertex Coordinates for all triangles in the TIN
    tri_index = tin.find_simplex(grids)         # triangle index containing each grid cell. value of -1 indicates "not found"
    
    inds = point_indices[tri_index] 
    inds = np.array([elem if tri_index[idx] >= 0 else np.full([3], np.nan, dtype=float) for idx,elem in enumerate(inds)]) # If a grids is not located in any triangles then its 'inds' are all NaN.
    
    tvc = tin_cord[tri_index] 
    tvc = np.array([elem if tri_index[idx] >= 0 else np.full([3,3], np.nan, dtype=float) for idx,elem in enumerate(tvc)]) # If a grids is not located in any triangles then its 'tvc' are all NaN.

    return (inds, tvc)


def get_partial_derivatives (tv, grids, tin_coeffs): # Compute partial derivatives of the Z function at interpolated points (Zp) w.r.t coords (x,y,z) of three triangle's vertices in TIN linear interpolation
    '''
    input:
        tv:    (m x 3 x 3) array of triangle vertices coordinates (centroid removed)
        grids: (m x 2) array of horizontal coordinates of grid cell centers
    output:
        TIN_par_deriv: (m x 9) array of partial derivatives for each grid cell 
    '''
    
    if tv.shape[0] != grids.shape[0]: sys.exit("get_partial_derivatives: 'tv' and 'grids' must have the same number of elements.")
    
    tic = time.time()
    tv = tv.reshape((len(tv), 9))
    x1, y1, z1 = tv[:,0], tv[:,1], tv[:,2]
    x2, y2, z2 = tv[:,3], tv[:,4], tv[:,5]
    x3, y3, z3 = tv[:,6], tv[:,7], tv[:,8]
    A, B, C, D = tin_coeffs[:,0], tin_coeffs[:,1], tin_coeffs[:,2], tin_coeffs[:,3]
    xp, yp = grids[:, 0], grids[:, 1]
    C2 = C ** 2
    E = ((xp * A) + (yp * B) + D)
    
    d = np.full((len(A), 9), np.nan, dtype=float)
    d[:,0] = (((y3 - y2) * E) + (((z2 - z3) * yp) + ((y2 * z3) - (y3 * z2))) * C) / C2
    d[:,3] = (((y1 - y3) * E) + (((z3 - z1) * yp) + ((y3 * z1) - (y1 * z3))) * C) / C2
    d[:,6] = (((y2 - y1) * E) + (((z1 - z2) * yp) + ((y1 * z2) - (y2 * z1))) * C) / C2
    d[:,1] = (((x2 - x3) * E) + (((z3 - z2) * xp) + ((x3 * z2) - (x2 * z3))) * C) / C2
    d[:,4] = (((x3 - x1) * E) + (((z1 - z3) * xp) + ((x1 * z3) - (x3 * z1))) * C) / C2
    d[:,7] = (((x1 - x2) * E) + (((z2 - z1) * xp) + ((x2 * z1) - (x1 * z2))) * C) / C2
    d[:,2] = (((y2 - y3) * xp) + ((x3 - x2) * yp) + ((x2 * y3) - (x3 * y2))) / C
    d[:,5] = (((y3 - y1) * xp) + ((x1 - x3) * yp) + ((x3 * y1) - (x1 * y3))) / C
    d[:,8] = (((y1 - y2) * xp) + ((x2 - x1) * yp) + ((x1 * y2) - (x2 * y1))) / C
 
    print('Calculatin partial derivatives of TIN Interpolation equations took {} seconds'.format(time.time() - tic))
    return d


def get_tin_coeffs(tv): # Estimate coefficients A, B, C, and D in triangular (TIN) linear interpolation
    '''    
    input:
        tv:     (m x 3 x 3) array of triangle vertices coordinates (centroid removed)
    output:
        coeffs: (t x 4) Coefficients A, B, C, and D to calculate TIN interpolation
    '''
    tic = time.time()

    x1, y1, z1 = tv[:, 0, 0], tv[:, 0, 1], tv[:, 0, 2]
    x2, y2, z2 = tv[:, 1, 0], tv[:, 1, 1], tv[:, 1, 2]
    x3, y3, z3 = tv[:, 2, 0], tv[:, 2, 1], tv[:, 2, 2]
    
    A = (y1 * z3) - (y1 * z2) + (y2 * z1) - (y2 * z3) + (y3 * z2) - (y3 * z1)
    B = (x1 * z2) - (x1 * z3) + (x2 * z3) - (x2 * z1) + (x3 * z1) - (x3 * z2)
    C = (x1 * y2) - (x1 * y3) + (x2 * y3) - (x2 * y1) + (x3 * y1) - (x3 * y2)
    D = (x1 * y2 * z3) - (x1 * y3 * z2) + (x2 * y3 * z1) - (x2 * y1 * z3) + (x3 * y1 * z2) - (x3 * y2 * z1)
    
    coeffs = np.column_stack((A, B, C, D))
    print('Calculatin partial derivatives of TIN coeffs took {} seconds'.format(time.time() - tic))
    
    return coeffs


def slice_data_idx(datalen, nbins):
    ''' 
    input:
        datalen:    The length of data
            Format: scalar
            Type:   scalar
        nbins:        The number of bins
            Format: scalar
            Type:   scalar
    output:
        slice_idx:    A list of list of indices of data after spliting
            Format:    [nbins] = [no of bins]
            Type:    List of list of indices
    '''
    aver, res = divmod(datalen, nbins)
    nums = [0] + [(aver+1) if bin<res else aver for bin in range(nbins)]    
    slice_idx = [list(np.arange(sum(nums[:ii]), sum(nums[:(ii+1)]))) for ii in np.arange(1,len(nums))]
    return slice_idx



def propagate_tin_error(tpu, inds, parderiv):
    tic = time.time()
    m = len(inds)
    n = len(tpu)
    grid_tpu = np.full((m), np.nan)
    sig = np.full((m, 9, 9), np.nan)
    z = np.zeros((3, 3))
    
    C = np.zeros((n, 3, 3))       # Cov matrix of point cloud
    C[:, 0, 0] = tpu[:, 0]
    C[:, 1, 1] = tpu[:, 3]
    C[:, 2, 2] = tpu[:, 5]
    C[:, 0, 1], C[:, 1, 0] = tpu[:, 1], tpu[:, 1]
    C[:, 0, 2], C[:, 2, 0] = tpu[:, 2], tpu[:, 2]
    C[:, 1, 2], C[:, 2, 1] = tpu[:, 4], tpu[:, 4]
    
    for i in range(m):
        if not np.isnan(parderiv[i, :]).any():
            j = inds[i].astype(int)
            sig = np.block([[C[j[0]], z, z], [z, C[j[1]], z], [z, z, C[j[2]]]])
            grid_tpu[i] = parderiv[i, ] @ sig @ np.transpose(parderiv[i, ])
    
    
    toctime = time.time() - tic
    print('Error propagation method 2 took {} seconds'.format(toctime))
    return grid_tpu



def Interpolate_TIN(grid, coeff, centroid):
    '''
    input:
        grid: (m x 2) array of horizontal coordinates of grid cell centers
        coeff: (m x 4) TIN interpolation coefficients [A, B, C, D]

    output:
            Zp: Interpolated height using TIN
                format: [ngrdpt, ]
                type: array
            
            Using the following formula:
                a = A / C
                b = B / C
                c = D / C
                Zp = (a * Xp) + (b * Yp) + c
    '''
    
    a = coeff[:, 0] / coeff[:, 2]
    b = coeff[:, 1] / coeff[:, 2]
    c = coeff[:, 3] / coeff[:, 2]
    
    Xp = grid[:, 0] #- centroid[0]
    Yp = grid[:, 1] #- centroid[1]
    
    Zp = (a * Xp) + (b * Yp) + c + centroid[2]
    return Zp




''' ----------------------------------------------------------------------- '''

# ifile = r'D:\Working\Luyen\Uncertainty\WaikoloaSample_res.h5'
# ofile = r'D:\Working\Luyen\Uncertainty\test.h5'

# h5ifile = h5py.File(ifile, 'r')
# ptcloud, sigm_COV = h5ifile['resptcloud'][()], h5ifile['res_sigm_COV'][()]
# h5ifile.close()

# mingrdX, grddX, maxgrdX = 464850, 0.1, 464900 
# mingrdY, grddY, maxgrdY = 121050, 0.1, 121100



''' ----------------------------------------------------------------------- '''

grid_size = 2.5


inlas = r'./data/input/fixed.laz'
p = pdal.Reader(inlas, use_eb_vlr=True).pipeline()
p.execute()
las = p.get_dataframe(0)

ptcloud = np.transpose(np.vstack([las['X'], las['Y'], las['Z']]))
sig_COV = np.transpose(np.vstack([las['VarianceX'], las['CovarianceXY'], 
                                  las['CovarianceXZ'], las['VarianceY'], 
                                  las['CovarianceXZ'], las['VarianceZ']]))

minx = p.metadata['metadata']['readers.las']['minx']
maxx = p.metadata['metadata']['readers.las']['maxx']
miny = p.metadata['metadata']['readers.las']['miny']
maxy = p.metadata['metadata']['readers.las']['maxy']


mingrdX, grddX, maxgrdX = np.ceil(minx), grid_size, np.floor(maxx)
mingrdY, grddY, maxgrdY = np.ceil(miny), grid_size, np.floor(maxy)


''' ----------------------------------------------------------------------- '''
grdX ,  grdY = np.arange(mingrdX, maxgrdX+grddX, grddX), np.arange(mingrdY, maxgrdY+grddY, grddY)
meshX, meshY = np.meshgrid(grdX, grdY)
grdpt = np.array(np.column_stack((meshX.ravel(), meshY.ravel())))
del grddX, maxgrdX, mingrdY, grddY, grdX, grdY



Z, dZ = TIN_lin_Err_Prop_many_MP_starmap2Async(ptcloud, sig_COV, grdpt)


ZZ = Z.reshape(meshX.shape)
fig = plt.figure()
plt.imshow(np.flipud(ZZ))

DZ = dZ.reshape(meshX.shape)
dz = np.sqrt(DZ)
fig = plt.figure()
plt.imshow(np.flipud(dz), vmin=np.nanmin(dz), vmax=0.6)




output_dem = r'./data/output/fixed_dem1.tif'
output_tpu = r'./data/output/fixed_tpu.tif'
ul = (mingrdX, maxgrdY)
pixelWidth = grid_size
pixelHeight = grid_size


write_raster(output_dem, ul, pixelWidth, pixelHeight, ZZ, 32610)
write_raster(output_tpu, ul, pixelWidth, pixelHeight, dz, 32610)

