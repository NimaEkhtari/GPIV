# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:07:46 2023

@author: nekhtari
"""

import numpy as np
import multiprocessing as mp
import h5py, sys
#import scipy.linalg
from scipy.spatial import Delaunay, KDTree
from os.path import dirname as dirname
from os.path import abspath as abspath
import rasterio
import time
import pdal
#from temp_funcs import TIN_lin_Err_Prop_many_MP_starmap2Async


def TIN_lin_Err_Prop_many_MP_starmap2Async(ptcloud, ptcloudsigmcov, grdpt, nprocs=mp.cpu_count()): # Error propagation from point cloud's uncertainties to interpolated pts by mp.starmap2Async
    grdtricord, grdtrisimplices = TIN_grdtricord_simplices(ptcloud, grdpt)
    parderiv = TIN_lin_par_deriv(grdtricord, grdpt)    # Partial derivatives of the Z fn at grdpt w.r.t coords of triangles' vertices from ptcloud:
                                                                                            # [ngrdpt, nparderiv=9] = [no of grid pts, number of partial derivatives (9: dzp/dx1, ..., dzp/dz3)]
    del grdtricord
    inp_list_idx = slice_data_idx(len(grdpt), nprocs)
    inp_grdtrisimplices, inp_parderiv = [[grdtrisimplices[item, :] for item in row] for row in inp_list_idx], [[parderiv[item] for item in row] for row in inp_list_idx]
    with mp.Pool(nprocs) as pool: results = pool.starmap_async(TIN_lin_Err_Prop_many, [(ptcloudsigmcov, inp_grdtrisimplices[ii], inp_parderiv[ii], ii+1) for ii in range(len(inp_list_idx))]).get()    # this calls pool.starmap_async, normally faster than pool.starmap
    grdpterr = [item for sublist in results for item in sublist]
    return np.array(grdpterr, dtype=float)


def TIN_grdtricord_simplices(ptcloud, grdpt): # Identify coordinates of vertices of TIN triangles formed from ptcloud where grdpt are located inside
    '''
    input:
        ptcloud:  Coords of pt. cloud
            Format: [nptclound, ncord=3 or 2] = [no of cloud pts, no of coordinates (3: X, Y, Z; 2: X, Y)]
            Type:   array
        grdpt:    Coords of grid pts
            Format: [ngrdpt, ncord=2] = [no of grid pts,  no of coordinates (2: X, Y)]
            Type:   array
    output:
        grdtricord: coordinates of vertices of triangles formed from ptcloud within which grdpt located
            Format: [ngrdpt, nvertex=3, ncord=2] = [no of grid pts, no of vertices=3, no of coordinates=2 (X,Y)]
            Type:   array
        grdtrisimplices: Indices of ptcloud corresponding to vertices of triangles formed from ptcloud within which grdpt located
            Format: [ngrdpt, nvertex=3] = [no of grid pts, no of vertices=3]
            Type:   array '''
    
    tri = Delaunay(ptcloud[:, 0:2]) # define a Delaunay network as an object by scipy.spatial.Delaunay
    trisimplices = tri.simplices    # Pt indices of all triangles' vertices: [ntri, nvertex=3] = [no of triangles, no of vertices (3)]
    tricord = ptcloud[tri.simplices] # Coords of vertices of all triangles: [ntri, nvertex=3, ncord=3] = [no of triangles, no of vertices (3), no of coordinates (3: X, Y, Z)].
    grdtriidx = tri.find_simplex(grdpt) # Triangles' indices in which grdpt is located: [ngrdpt, ] = [no of grid pts, ]. '-1' = the point does not locate inside any triangle.
    del tri
    grdtrisimplices = trisimplices[grdtriidx] # Pt indices of only triangles in which grdpt is locatied: [ngrdpt, nvertex=(3)] = [no of grid pts, no of vertices (3)].
    grdtrisimplices = np.array([elem if grdtriidx[idx] >= 0 else np.full([3], np.nan, dtype=float) for idx,elem in enumerate(grdtrisimplices)]) # If a grdpt is not located in any triangles then its 'grdtrisimplices' are all NaN.
    del trisimplices
    grdtricord = tricord[grdtriidx] # Coords of vertices of only triangles in which grdpt is located: [ngrdpt, nvertex=3, ncord=3] = [no of grid pts., no of vertices (3), no or coordinates (3: X, Y, Z)]].
    grdtricord = np.array([elem if grdtriidx[idx] >= 0 else np.full([3,3], np.nan, dtype=float) for idx,elem in enumerate(grdtricord)]) # If a grdpt is not located in any triangles then its 'grdtricord' are all NaN.
    return (grdtricord, grdtrisimplices)


def TIN_lin_par_deriv   (grdtricord, grdpt): # Compute partial derivatives of the Z function at interpolated points (Zp) w.r.t coords (x,y,z) of three triangle's vertices in TIN linear interpolation
    '''
    input:
        grdtricord: Coords of triangles' vertices
            Format: [ngrdpt, nvertex=3, ncord=3] = [no of grid pts, no of vertices (3), no of coordinates (3: X, Y, Z)]
            Type:   array
        grdpt: Coords of grid pts
            Format: [ngrdpt, ncord=2] = [no of grid pts,  no of coordinates (2: X, Y)]
            Type:   array
    output:
        TIN_par_deriv:
            Format: [ngrdpt, noparderiv=9] = [no of grid pts, number of partial derivatives (9: dzp/dx1, ..., dzp/dz3)]
            Type:   array '''
    if grdtricord.shape[0] != grdpt.shape[0]: sys.exit("TIN_lin_par_deriv: 'grdtricord' and 'grdpt' must have the same number of elements.")
    TIN_coefs = TIN_lin_coefs_est(grdtricord) # [ngrdpt, ncoefs=4] = [no of grid pts, no of coefficients (4: A, B, C, D)]
    tic, TIN_par_deriv = time.time(), []
    for idx, grdptcord in enumerate(grdpt):
        sub_par_deriv = np.full(9, np.nan, dtype=float) # if the considered grid pt is NOT located in any TIN triangles then all 'grdtricord' corresponding to this pt are NaN, and all 'sub_par_deriv' are also NaN
        if not np.isnan(grdtricord[idx]).any(): # if the considered grid pt is located in a TIN triangle then do as below
            x1, y1, z1, x2, y2, z2, x3, y3, z3 = grdtricord[idx,0,0], grdtricord[idx,0,1], grdtricord[idx,0,2], grdtricord[idx,1,0], grdtricord[idx,1,1], grdtricord[idx,1,2], grdtricord[idx,2,0], grdtricord[idx,2,1], grdtricord[idx,2,2]
            A, B, C, D = TIN_coefs[idx,0], TIN_coefs[idx,1], TIN_coefs[idx,2], TIN_coefs[idx,3]
            xp, yp = grdptcord[0], grdptcord[1]
            sub_par_deriv[0] = 1/(C**2)*((y3-y2)*xp*A+(y3-y2)*yp*B+((z2-z3)*yp+(y2*z3-y3*z2))*C+(y3-y2)*D)
            sub_par_deriv[1] = 1/(C**2)*((x2-x3)*xp*A+(x2-x3)*yp*B+((z3-z2)*xp+(x3*z2-x2*z3))*C+(x2-x3)*D)
            sub_par_deriv[2] = 1/C*((y2-y3)*xp+(x3-x2)*yp+(x2*y3-x3*y2))
            sub_par_deriv[3] = 1/(C**2)*((y1-y3)*xp*A+(y1-y3)*yp*B+((z3-z1)*yp+(y3*z1-y1*z3))*C+(y1-y3)*D)
            sub_par_deriv[4] = 1/(C**2)*((x3-x1)*xp*A+(x3-x1)*yp*B+((z1-z3)*xp+(x1*z3-x3*z1))*C+(x3-x1)*D)
            sub_par_deriv[5] = 1/C*((y3-y1)*xp+(x1-x3)*yp+(x3*y1-x1*y3))
            sub_par_deriv[6] = 1/(C**2)*((y2-y1)*xp*A+(y2-y1)*yp*B+((z1-z2)*yp+(y1*z2-y2*z1))*C+(y2-y1)*D)
            sub_par_deriv[7] = 1/(C**2)*((x1-x2)*xp*A+(x1-x2)*yp*B+((z2-z1)*xp+(x2*z1-x1*z2))*C+(x1-x2)*D)
            sub_par_deriv[8] = 1/C*((y1-y2)*xp+(x2-x1)*yp+(x1*y2-x2*y1))
        TIN_par_deriv.append(sub_par_deriv)

        toctime = time.time() - tic
        print('Taking partial derivatives took {} seconds'.format(toctime))
    return np.array(TIN_par_deriv, dtype=float)


def TIN_lin_coefs_est(tricord): # Estimate coefficients A, B, C, and D in triangular (TIN) linear interpolation
    '''    
    input:
        tricord: Coords of triangles' vertices
            Format: [ntri, nvertex=3, ncord=3] = [no of triangles, no of vertices (3), no of coordinates (3: X, Y, Z)]
            Type:   array
    output:
        TIN_coefs: Coefficients A, B, C, and D
            Format: [ntri, ncoefs=4] = [no of triangles, no of coefficients (4: A, B, C, D)]
            Type:   array'''
    TIN_coefs = []
    for elem in tricord:
        if np.isnan(elem).any(): # If the considered pt is not located in any TIN triangles then all 'tricord' corresponding to this pt is NaN, and all 'TIN_coefs' are also NaN
            TIN_coefs.append(np.full(4, np.nan, dtype=float))
        else:
            A = elem[0,1]*elem[2,2]-elem[0,1]*elem[1,2]+elem[1,1]*elem[0,2]-elem[1,1]*elem[2,2]+elem[2,1]*elem[1,2]-elem[2,1]*elem[0,2]
            B = elem[0,0]*elem[1,2]-elem[0,0]*elem[2,2]+elem[1,0]*elem[2,2]-elem[1,0]*elem[0,2]+elem[2,0]*elem[0,2]-elem[2,0]*elem[1,2]
            C = elem[0,0]*elem[1,1]-elem[0,0]*elem[2,1]+elem[1,0]*elem[2,1]-elem[1,0]*elem[0,1]+elem[2,0]*elem[0,1]-elem[2,0]*elem[1,1]
            D = elem[0,0]*elem[1,1]*elem[2,2]-elem[0,0]*elem[2,1]*elem[1,2]+elem[1,0]*elem[2,1]*elem[0,2]-elem[1,0]*elem[0,1]*elem[2,2]+elem[2,0]*elem[0,1]*elem[1,2]-elem[2,0]*elem[1,1]*elem[0,2]
            TIN_coefs.append([A, B, C, D])
    return np.array(TIN_coefs, dtype=float)


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


def TIN_lin_Err_Prop_many(ptcloudsigmcov, grdtrisimplices, parderiv, worker=1): # Estimate propagated uncertainties of interpolated points given 1) ptcloud sigma, 2) indices of ptcloud corresponding to triangles' vertices, 3) partial derivatives of the heights of interpolated pts w.r.t triangles' vertices' coordinate
    '''
    input:
        ptcloudsigmcov: Variance-Coviarance of pt cloud uncertainties
            Format: [nptcloud, nvarcov=6] = [no of cloud pts, no of var-cov values (6: sig_x^2, sig_xy, sig_xz, sig_y^2, sig_yz, sig_z^2]
            type:   array
        grdtrisimplices:    Indices of ptcloud corresponding to vertices of triangles
            Format:    [nintpt, nvertex=(3)] = [no of interpolated pts, no of vertices (3)].
            type:    array
        parderiv:    Partial derivatives of the height of interpolated points w.r.t triangles' vertices' coordinates
            Format:    [nintpt, nparderiv=9] = [no of interpolated pts, number of partial derivatives (9: dzp/dx1, ..., dzp/dz3)]
            type:    array
    output:
        interr: Propagated uncertainties in the z component at the interpolated pts.
            Format: [nintpt, ] = [no of interpolated pts, ]
            Type:   array '''
    if type(grdtrisimplices) is list: grdtrisimplices = np.array(grdtrisimplices)
    if type(parderiv)        is list:        parderiv = np.array(       parderiv)
    tic, interr = time.time(), []
    for idx in range(len(parderiv)):
        if np.isnan(parderiv[idx,:]).any():
            interr.append(np.nan)
        else:
            sigm_COV = np.zeros([9,9], dtype=float)
            for ii in range(3): # Run thru three triangle's vertices
                sigm_COV_ii = np.zeros([3, 3], dtype=float)
                triu = np.triu_indices(sigm_COV_ii.shape[0]) # The indices for the upper-triangle of sigm_COV_ii
                sigm_COV_ii[triu] = ptcloudsigmcov[int(grdtrisimplices[idx, ii])] # Assign 'ptcloudsigmcov' to the upper-triangle of sigm_COV_ii
                sigm_COV_ii = sigm_COV_ii + sigm_COV_ii.T - np.diag(np.diag(sigm_COV_ii)) # Make a symmetric array
                sigm_COV[ii*3:ii*3+3,ii*3:ii*3+3] = sigm_COV_ii
            interr.append(parderiv[idx,:].dot(sigm_COV).dot(np.transpose(parderiv[idx,:])))
    
    toctime = time.time() - tic
    print('Error propagation took {} seconds'.format(toctime))
    return np.array(interr, dtype=float)





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
del mingrdX, grddX, maxgrdX, mingrdY, grddY, maxgrdY, grdX, grdY, meshX, meshY



Z = TIN_lin_Err_Prop_many_MP_starmap2Async(ptcloud, sig_COV, grdpt)

out_tiff = r'./tpu.tif'
new_dataset = rasterio.open(out_tiff, 'w', driver='GTiff', height=Z.shape[0],
    width=Z.shape[1], count=1, dtype=Z.dtype, crs='+proj=latlong', transform=transform)

new_dataset.write(Z, 1)
new_dataset.close()

