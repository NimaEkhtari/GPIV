# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:33:30 2023

@author: nekhtari
"""

import multiprocessing as mp
from cmath import nan
import numpy as np
import scipy.linalg
from scipy.spatial import Delaunay, KDTree
import sys, time, inspect

def TIN_lin_Err_Prop_many_MP_starmap2Async(ptcloud, ptcloudsigmcov, grdpt, verbose=False, verbosetab='', verboseprct=1, nprocs=mp.cpu_count()): # Error propagation from point cloud's uncertainties to interpolated pts by mp.starmap2Async
    '''    inputs & outputs are the same as those in TIN_lin_Err_Prop_many_Serial '''
    if nprocs < 2: raise ValueError('\r\n' + verbosetab + __name__ + '.' + inspect.currentframe().f_code.co_name + ": The number of workers 'nprocs' must be at least 2 \n")
    if nprocs > len(grdpt): raise ValueError('\r\n' + verbosetab + __name__ + '.' + inspect.currentframe().f_code.co_name + ": The number of workers 'nprocs' must be smaller than or equal to " + str(len(grdpt)) + " \n")
    grdtricord, grdtrisimplices = TIN_grdtricord_simplices(ptcloud, grdpt, verbose, verbosetab)
    if verbose: print(verbosetab+"Estimate partial derivatives of the heights of interpolated points w.r.t. coordinates of triangles' vertices.")
    parderiv = TIN_lin_par_deriv(grdtricord, grdpt, verbose, verbosetab+'\t', verboseprct)    # Partial derivatives of the Z fn at grdpt w.r.t coords of triangles' vertices from ptcloud:
                                                                                            # [ngrdpt, nparderiv=9] = [no of grid pts, number of partial derivatives (9: dzp/dx1, ..., dzp/dz3)]
    del grdtricord
    if verbose: print(verbosetab+"Start computing propagated uncertainties with multiprocessing.starmapAsync")
    inp_list_idx = slice_data_idx(len(grdpt), nprocs)
    inp_grdtrisimplices, inp_parderiv = [[grdtrisimplices[item, :] for item in row] for row in inp_list_idx], [[parderiv[item] for item in row] for row in inp_list_idx]
    with mp.Pool(nprocs) as pool: results = pool.starmap_async(TIN_lin_Err_Prop_many, [(ptcloudsigmcov, inp_grdtrisimplices[ii], inp_parderiv[ii], verbose, verbosetab, verboseprct, ii+1) for ii in range(len(inp_list_idx))]).get()    # this calls pool.starmap_async, normally faster than pool.starmap
    grdpterr = [item for sublist in results for item in sublist]
    return np.array(grdpterr, dtype=float)

def TIN_lin_Err_Prop_many_Serial          (ptcloud, ptcloudsigmcov, grdpt, verbose=False, verbosetab='', verboseprct=1): # Error propagation from point cloud's uncertainties to interpolated pts by the Serial approach
    '''
    input:
        ptcloud:  Coords of pt. cloud
            Format: [nptclound, ncord=3] = [no of cloud pts, no of coordinates (3: X, Y, Z)]
            Type:   array
        ptcloudsigmcov: Variance-Coviarance of pt cloud uncertainties
            Format: [nptcloud, nvarcov=6] = [no of cloud pts, no of var-cov values (6: sig_x^2, sig_xy, sig_xz, sig_y^2, sig_yz, sig_z^2]
            type:   array
        grdpt:    Coords of grid pts
            Format: [ngrdpt, ncord=2] = [no of grid pts,  no of coordinates (2: X, Y)]
            Type:   array
    output:
        grdpterr: Propagated uncertainties in the z component at the gridded pts.
            Format: [ngrdpt, ] = [no of grid pts, ]
            Type:   array '''
    grdtricord, grdtrisimplices = TIN_grdtricord_simplices(ptcloud, grdpt, verbose, verbosetab)
    if verbose: print(verbosetab+"Estimate partial derivatives of the heights of interpolated points w.r.t. coordinates of triangles' vertices.")
    parderiv = TIN_lin_par_deriv(grdtricord, grdpt, verbose, verbosetab+'\t', verboseprct)    # Partial derivatives of the Z fn at grdpt w.r.t coords of triangles' vertices from ptcloud:
                                                                                        # [ngrdpt, nparderiv=9] = [no of grid pts, number of partial derivatives (9: dzp/dx1, ..., dzp/dz3)]
    del grdtricord
    if verbose: print(verbosetab+"Start computing propagated uncertainties with serial computing")
    grdpterr = TIN_lin_Err_Prop_many(ptcloudsigmcov, grdtrisimplices, parderiv, verbose, verbosetab, verboseprct)
    return np.array(grdpterr, dtype=float)

def TIN_lin_Err_Prop_many(ptcloudsigmcov, grdtrisimplices, parderiv, verbose=False, verbosetab='', verboseprct=1, worker=1): # Estimate propagated uncertainties of interpolated points given 1) ptcloud sigma, 2) indices of ptcloud corresponding to triangles' vertices, 3) partial derivatives of the heights of interpolated pts w.r.t triangles' vertices' coordinate
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
    tic, intvlprct, interr = time.time(), max(1, round(float(len(parderiv))*verboseprct/100)), []
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
        if verbose:
            if (idx + 1) == 1 or (idx + 1) % intvlprct == 0 or (idx + 1) == len(parderiv):
                prgrs_prct, prgrs_runtime = float(idx+1)/len(parderiv)*100, time.time() - tic
                prgrs_str = verbosetab + 'Worker: ' + str(worker) + '; Estimated pt no: ' + str(idx + 1) + '/' + str(len(parderiv)) + ' [' + str(int(round(prgrs_prct))) + '%]'
                f = prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=True) if (idx + 1) == len(parderiv) else prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=False)
    return np.array(interr, dtype=float)

def TIN_grdtricord_simplices(ptcloud, grdpt, verbose=False, verbosetab=''): # Identify coordinates of vertices of TIN triangles formed from ptcloud where grdpt are located inside
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
    if verbose: print(verbosetab+"Generate a Delaunay network")
    tri = Delaunay(ptcloud[:, 0:2]) # define a Delaunay network as an object by scipy.spatial.Delaunay
    if verbose: print(verbosetab+"Search indices of ptcloud corresponding to triangles' vertices.")
    trisimplices = tri.simplices    # Pt indices of all triangles' vertices: [ntri, nvertex=3] = [no of triangles, no of vertices (3)]
    if verbose: print(verbosetab+"Find coordinates of trangles' vertices")
    tricord = ptcloud[tri.simplices] # Coords of vertices of all triangles: [ntri, nvertex=3, ncord=3] = [no of triangles, no of vertices (3), no of coordinates (3: X, Y, Z)].
    if verbose: print(verbosetab+"Filter only triangles' in which grid points are located inside")
    grdtriidx = tri.find_simplex(grdpt) # Triangles' indices in which grdpt is located: [ngrdpt, ] = [no of grid pts, ]. '-1' = the point does not locate inside any triangle.
    del tri
    if verbose: print(verbosetab+"Search indices of ptcloud of vertices of only triangles in which grid points are located inside")
    grdtrisimplices = trisimplices[grdtriidx] # Pt indices of only triangles in which grdpt is locatied: [ngrdpt, nvertex=(3)] = [no of grid pts, no of vertices (3)].
    grdtrisimplices = np.array([elem if grdtriidx[idx] >= 0 else np.full([3], np.nan, dtype=float) for idx,elem in enumerate(grdtrisimplices)]) # If a grdpt is not located in any triangles then its 'grdtrisimplices' are all NaN.
    del trisimplices
    if verbose: print(verbosetab+"Search coordinates of vertices of only triangles in which grid points are located inside")
    grdtricord = tricord[grdtriidx] # Coords of vertices of only triangles in which grdpt is located: [ngrdpt, nvertex=3, ncord=3] = [no of grid pts., no of vertices (3), no or coordinates (3: X, Y, Z)]].
    grdtricord = np.array([elem if grdtriidx[idx] >= 0 else np.full([3,3], np.nan, dtype=float) for idx,elem in enumerate(grdtricord)]) # If a grdpt is not located in any triangles then its 'grdtricord' are all NaN.
    return (grdtricord, grdtrisimplices)

def TIN_lin_par_deriv   (grdtricord, grdpt, verbose=False, verbosetab='', verboseprct=1): # Compute partial derivatives of the Z function at interpolated points (Zp) w.r.t coords (x,y,z) of three triangle's vertices in TIN linear interpolation
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
    if verbose: print(verbosetab+"Estimate coefficients A, B, C, D of TIN linear interpolation")
    TIN_coefs = TIN_lin_coefs_est(grdtricord) # [ngrdpt, ncoefs=4] = [no of grid pts, no of coefficients (4: A, B, C, D)]
    if verbose: print(verbosetab+"Start computing partial derivatives")
    tic, intvlprct, TIN_par_deriv = time.time(), max(1, round(float(len(grdpt))*verboseprct/100)), []
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
        if verbose:
            if (idx + 1) == 1 or (idx + 1) % intvlprct == 0 or (idx + 1) == len(grdpt):
                prgrs_prct, prgrs_runtime = float(idx + 1) / len(grdpt) * 100, time.time() - tic
                prgrs_str = verbosetab + 'Estimated triangle no: ' + str(idx + 1) + '/' + str(len(grdpt)) + ' [' + str(int(round(prgrs_prct))) + '%]'
                f = prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=True) if (idx + 1) == len(grdpt) else prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=False)
    return np.array(TIN_par_deriv, dtype=float)

def TIN_lin_interp(ptcloud, grdpt, demeaned=False, verbose=False, verbosetab='', verboseprct=1): # Interpolate Z for grid pts from point cloud based on a TIN with linear interpolation
    '''
    input:
        ptcloud:  Coords of pt. cloud
            Format: [nptclound, ncord=3] = [no of cloud pts, no of coordinates (3: X, Y, Z)]
            Type:   array
        grdpt:    Coords of grid pts
            Format: [ngrdpt, ncord=2] = [no of grid pts,  no of coordinates (2: X, Y)] => Z will be linearly interpolated
            Type:   array
    output:
        interpz:
            Format: [ngrdpt, ] = [no of grid pts, ]
            Type:   array '''
    if demeaned: # demeaned is to avoid computational error with large number
        meanX, meanY = np.mean([np.max(ptcloud[:, 0]), np.min(ptcloud[:, 0])]), np.mean([np.max(ptcloud[:, 1]), np.min(ptcloud[:, 1])])
        #meanX, meanY = np.mean(ptcloud[:, 0]), np.mean(ptcloud[:, 1])
        ptcloud = np.array(list(map(lambda X: [X[0]-meanX, X[1]-meanY, X[2]], ptcloud.tolist())))
        grdpt = np.array(list(map(lambda X: [X[0]-meanX, X[1]-meanY], grdpt.tolist())))
    if verbose: print(verbosetab + 'Generate a Delaunay network')
    tri = Delaunay(ptcloud[:, 0:2]) # define a Delaunay network as an object by scipy.spatial.Delaunay
    tricord = ptcloud[tri.simplices] # Coords of vertices of all triangles: [ntri, nvertex=3, ncord=3] = [no of triangles, no of vertices (3), no of coordinates (3: X, Y, Z)].
    grdtriidx = tri.find_simplex(grdpt) # Triangles' indices in which grdpt is located: [ngrdpt, ] = [no of grid pts, ]. '-1' = the point does not locate inside any triangle.
    del tri
    if verbose: print(verbosetab + 'Extract coordinates of vertices of only triangles in which grdpt is located inside')
    grdtricord = tricord[grdtriidx] # Coords of vertices of only triangles in which grdpt is located: [ngrdpt, nvertex=3, ncord=3] = [no of grid pts., no of vertices (3), no or coordinates (3: X, Y, Z)]].
    grdtricord = np.array([elem if grdtriidx[idx] >= 0 else np.full([3,3], np.nan, dtype=float) for idx,elem in enumerate(grdtricord)]) # If a grdpt is not located in any triangles then its 'grdtricord' are all NaN.'''
    del tricord, grdtriidx
    if verbose: print(verbosetab + 'Estimate TIN linear coefficents of Delaunay triangles')
    TIN_coefs = TIN_lin_coefs_est(grdtricord) # [ngrdpt, ncoefs=4] = [no of grid pts, no of coefficients (4: A, B, C, D)]
    if verbose: print(verbosetab + 'Conduct the interpolation by linear TIN')
    interpz = [] # Interpolated Z of grdpt
    tic, intvlprct = time.time(), round(float(len(grdpt))*verboseprct/100)
    for idx, grdptcord in enumerate(grdpt):
        result = np.nan
        if not np.isnan(TIN_coefs[idx]).any():
            A, B, C, D = TIN_coefs[idx]
            xp, yp = grdptcord
            result = A/C*xp + B/C*yp + D/C
        interpz.append(result)
        if verbose:
            if (idx + 1) == 1 or (idx + 1) % intvlprct == 0 or (idx + 1) == len(grdpt):
                prgrs_prct, prgrs_runtime = float(idx + 1) / len(grdpt) * 100, time.time() - tic
                prgrs_str = verbosetab + '\tEstimated pt no: ' + str(idx + 1) + '/' + str(len(grdpt)) + ' [' + str(int(round(prgrs_prct))) + '%]'
                f = prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=True) if (idx + 1) == len(grdpt) else prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=False)
    return np.array(interpz, dtype=float)

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

def cloud_roughness_density_many_MP_starmap2Async(ptcloud, grdpt, grdptflag, wwid, min_sample_no=0, nondeviation=False, deviation=True, method='KDTree', verbose=False, verbosetab='', verboseprct=1, nprocs=mp.cpu_count()): # Call cloud_del_near_pt_dist_many with multiprocessing.
    ''' input & output are the same as those in cloud_roughness_density_many_KDTree '''
    if nprocs < 2: raise ValueError('\r\n' + verbosetab + __name__ + '.' + inspect.currentframe().f_code.co_name + ": The number of workers 'nprocs' must be at least 2 \n")
    if nprocs > len(grdpt): raise ValueError('\r\n' + verbosetab + __name__ + '.' + inspect.currentframe().f_code.co_name + ": The number of workers 'nprocs' must be smaller than or equal to " + str(len(grdpt)) + " \n")
    inp_list_idx = slice_data_idx(len(grdpt), nprocs)
    inp_lists, inp_flags = [[grdpt[item, :] for item in row] for row in inp_list_idx], [[grdptflag[item] for item in row] for row in inp_list_idx]
    if verbose: print(verbosetab +'Method used: ' + method)
    if method.lower() == 'kdtree':
        tree = KDTree(ptcloud[:,:2])
        with mp.Pool(nprocs) as pool: results = pool.starmap_async(cloud_roughness_density_many_KDTree, [(ptcloud, inp_lists[ii], inp_flags[ii], wwid, min_sample_no, nondeviation, deviation, verbose, verbosetab, verboseprct, ii+1, tree) for ii in range(len(inp_lists))]).get()
    elif method.lower() == 'dxdy':
        with mp.Pool(nprocs) as pool: results = pool.starmap_async(cloud_roughness_density_many_dXdY,   [(ptcloud, inp_lists[ii], inp_flags[ii], wwid, min_sample_no, nondeviation, deviation, verbose, verbosetab, verboseprct, ii+1)       for ii in range(len(inp_lists))]).get()
    else:
        raise ValueError('\r\n' + verbosetab + __name__ + '.' + inspect.currentframe().f_code.co_name + ": The method set must be one of those in the list: " + str(['kdtree', 'faiss', 'cdist']) + " \n")
    grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts = np.array([item for sublist in results for item in sublist[0]]), np.array([item for sublist in results for item in sublist[1]]), np.array([item for sublist in results for item in sublist[2]])
    return grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts

def cloud_roughness_density_many_KDTree          (ptcloud, grdpt, grdptflag, wwid, min_sample_no=0, nondeviation=False, deviation=True, verbose=False, verbosetab='', verboseprct=1, worker=1, tree=None): # Call cloud_roughness_density_one_KDTree to apply with many grdpt
    ''' This calls cloud_roughness_density_one_KDTree to applied with many grdpt    
    input:
        grdpt:    Coordinates of the point dataset of which the terrain roughness and point density will be estimated
            Format: [npts, ncord] = [no of points, no of coordinates (2: [X, Y] or 3: [X, Y, Z])]
            Type:   array or list (list will be converted to array)
        grdptflag:    A boolen (True/False) array showing whether or not a point is estimated for its terrain roughness and number of points within a moving window
            Format:    [npts,] = [no of points]
            Type:    Boolean
        others: ptcloud, wwid, min_sample_no, nondeviation, deviation: the same as those in cloud_roughness_density_one
    output:
        grdpt_sigZ, grdpt_sigZ_dev:    The same as those in cloud_roughness_density_one
    Important notes:
        If the number point situated within the moving window smaller than        'min_sample_no'  then 'grdpt_sigZ_dev' = NaN
        If the number point situated within the moving window smaller than max(3, 'min_sample_no') then     'grdpt_sigZ' = NaN
    '''
    if tree == None: raise ValueError('\r\n' + __name__ + '.' + inspect.currentframe().f_code.co_name + ": KDTree must be provided \n")
    if type(grdpt) is list: # convert from list to numpy array
        grdpt = np.array(grdpt)
    grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts = [], [], [] # terrain rouhghness, no of points situated within the moving circle
    tic, intvlprct = time.time(), max(1, round(float(len(grdpt))*verboseprct/100))
    for ii, item in enumerate(grdpt):
        grdpt_sigZ.append(np.nan), grdpt_sigZ_dev.append(np.nan), grdpt_pts.append(np.nan)    # in case (grdptflag[ii] = False)
        if grdptflag[ii]: grdpt_sigZ[-1], grdpt_sigZ_dev[-1], grdpt_pts[-1] = cloud_roughness_density_one_KDTree(ptcloud, item, wwid, min_sample_no, nondeviation, deviation, tree)
        if verbose:
            if (ii + 1) == 1 or (ii + 1) % intvlprct == 0 or (ii + 1) == len(grdpt):
                prgrs_prct, prgrs_runtime = float(ii + 1) / len(grdpt) * 100, time.time() - tic
                prgrs_str = verbosetab + 'Worker: ' + str(worker) + '; Estimated pt no: ' + str(ii + 1) + '/' + str(len(grdpt)) + ' [' + str(int(round(prgrs_prct))) + '%]'
                f = prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=True) if (ii + 1) == len(grdpt) else prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=False)
    grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts = np.array(grdpt_sigZ), np.array(grdpt_sigZ_dev), np.array(grdpt_pts)
    return grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts

def cloud_roughness_density_one_KDTree           (ptcloud, grdpt, radius, min_sample_no=0, nondeviation=False, deviation=True, tree=None): # Estimation of the terrain roughness (i.e., std dev of Z and/or std dev of residual of Z) and the number of point for ONE grdpt from ptcloud within a circle by KDTree
    ''' A circle is centred at grdpt then all ptcloud located within the circle are extracted and used to est terrain roughness and point density.
    input:
        ptcloud:    Coordinates of the point cloud dataset
            Format: [npts, ncord = 3] = [no of points, no of coordinates (3: [X, Y, Z])]
            Type:   array
        grdpt:        Coordinates of grdpt of which terrain roughness and point density is estimated
            Format: [ncord, ] = [no of coordinates (2: [X, Y] or 3: [X, Y, Z]), ]
            Type:   array
        tree:        KDTree object
            Format:    Object
            Type:    Object
        radius:        Radius of the moving circle
            Format:    Scalar
            Type:    Scalar
        min_sample_no:    Minimum no of samples found located within moving circle. Set NaN if the no found smaller than this value
            Format:    Scalar
            Type:    Scalar
        nondeviation:    Terrain roughness defined by std dev of Z is esimated if nondeviation == True                    
            Format:    Scalar
            Type:    Boolean
        deviation:    Terrain roughness defined by std dev of the residual of Z is esimated if deviation == True
                    If deviation == True then a least-squares plane is generated first from ptcloud located within the moving window, then deviation of Z is computed as the perp distance between cloud points and the LS-fitted plane
            Format:    Scalar
            Type:    Boolean        
    output:
        grdpt_sigZ:    Terrain roughness defined by std dev of Z of the resampled dataset computed at the moving circle of wwid size
            Format:    scalar
            Type:    scalar
        grdpt_sigZ_dev:    Terrain roughness defined by std dev of deviation of Z of the resampled dataset computed at the moving circle of wwid size
            Format:    scalar
            Type:    scalar        
        grdpt_pts:    The number of points found within the moving circle
            Format:    scalar
            Type:    scalar
    Important notes:
        If the number point situated within the moving circle smaller than        'min_sample_no'  then 'grdpt_sigZ_dev' = NaN
        If the number point situated within the moving circle smaller than max(3, 'min_sample_no') then     'grdpt_sigZ' = NaN
    '''
    if tree == None: raise ValueError('\r\n' + __name__ + '.' + inspect.currentframe().f_code.co_name + ": KDTree must be provided \n")
    lstidx = tree.query_ball_point(grdpt[:2], radius) # This uses KDTree to search for points within a circle: use 'workers=...' for parallel running
    grdpt_sigZ, grdpt_sigZ_dev = nan, nan # grdpt_sigZ = nan if len(lstidx) < min_sample_no; grdpt_sigZ_dev = nan if len(lstidx) < max(3, min_sample_no)
    if nondeviation and len(lstidx) >= max(1, min_sample_no): grdpt_sigZ = np.std(ptcloud[lstidx,2])
    if deviation    and len(lstidx) >= max(3, min_sample_no): # at least max of (3, min_sample_no points) points are needed to generated linear-fit plane then est deviations
        subptcloud = ptcloud[lstidx]
        COEF = cloud_plane_coef_bestfit_min_z(subptcloud) # coef of the eq. of a LS-fit plane
        subptcloud_dev = []
        for item in subptcloud: subptcloud_dev.append(pt_plane_dist(COEF, item))
        grdpt_sigZ_dev = np.std(np.array(subptcloud_dev))    
    grdpt_pts = len(lstidx)
    return (grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts)

def cloud_roughness_density_many_dXdY            (ptcloud, grdpt, grdptflag, wwid, min_sample_no=0, nondeviation=False, deviation=True, verbose=False, verbosetab='', verboseprct=1, worker=1): # Call cloud_roughness_density_one_dXdY  with many grdpt
    ''' This calls cloud_roughness_density_one_dXdY to applied with many grdpt    
    input:
        grdpt:    Coordinates of the point dataset of which the terrain roughness and point density will be estimated
            Format: [npts, ncord] = [no of points, no of coordinates (2: [X, Y] or 3: [X, Y, Z])]
            Type:   array or list (list will be converted to array)
        grdptflag:    A boolen (True/False) array showing whether or not a point is estimated for its terrain roughness and number of points within a moving window
            Format:    [npts,] = [no of points]
            Type:    Boolean
        others: ptcloud, wwid, min_sample_no, nondeviation, deviation: the same as those in cloud_roughness_density_one
    output:
        grdpt_sigZ, grdpt_sigZ_dev:    The same as those in cloud_roughness_density_one
    Important notes:
        If the number point situated within the moving window smaller than        'min_sample_no'  then 'grdpt_sigZ_dev' = NaN
        If the number point situated within the moving window smaller than max(3, 'min_sample_no') then     'grdpt_sigZ' = NaN
    '''
    if type(grdpt) is list: # convert from list to numpy array
        grdpt = np.array(grdpt)
    grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts = [], [], [] # terrain rouhghness, no of points situated within moving windows
    tic, intvlprct = time.time(), max(1, round(float(len(grdpt))*verboseprct/100))
    for ii, item in enumerate(grdpt):
        grdpt_sigZ.append(np.nan), grdpt_sigZ_dev.append(np.nan), grdpt_pts.append(np.nan)    # in case (grdptflag[ii] = False)
        if grdptflag[ii]: grdpt_sigZ[-1], grdpt_sigZ_dev[-1], grdpt_pts[-1] = cloud_roughness_density_one_dXdY(ptcloud, item, wwid, min_sample_no, nondeviation, deviation)
        if verbose:
            if (ii + 1) == 1 or (ii + 1) % intvlprct == 0 or (ii + 1) == len(grdpt):
                prgrs_prct, prgrs_runtime = float(ii + 1) / len(grdpt) * 100, time.time() - tic
                prgrs_str = verbosetab + 'Worker: ' + str(worker) + '; Estimated pt no: ' + str(ii + 1) + '/' + str(len(grdpt)) + ' [' + str(int(round(prgrs_prct))) + '%]'
                f = prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=True) if (ii + 1) == len(grdpt) else prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=False)
    grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts = np.array(grdpt_sigZ), np.array(grdpt_sigZ_dev), np.array(grdpt_pts)
    return grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts

def cloud_roughness_density_one_dXdY             (ptcloud, grdpt,   wwid, min_sample_no=0, nondeviation=False, deviation=True): # Estimation of the terrain roughness (i.e., std dev of Z and/or std dev of residual of Z) and the number of point for ONE grdpt from ptcloud within a moving window.
    ''' A window is centred at grdpt then all ptcloud located within the window are extracted and used to est terrain roughness and point density.
    input:
        ptcloud:    Coordinates of the point cloud dataset
            Format: [npts, ncord = 3] = [no of points, no of coordinates (3: [X, Y, Z])]
            Type:   array
        grdpt:    Coordinates of grdpt of which terrain roughness and point density is estimated
            Format: [ncord, ] = [no of coordinates (2: [X, Y] or 3: [X, Y, Z]), ]
            Type:   array
        wwid:        Size of the moving window
            Format:    Scalar
            Type:    Scalar
        min_sample_no:    Minimum no of samples found located within moving windows. Set NaN if the no found smaller than this value
            Format:    Scalar
            Type:    Scalar
        nondeviation:    Terrain roughness defined by std dev of Z is esimated if nondeviation == True                    
            Format:    Scalar
            Type:    Boolean
        deviation:    Terrain roughness defined by std dev of the residual of Z is esimated if deviation == True
                    If deviation == True then a least-squares plane is generated first from ptcloud located within the moving window, then deviation of Z is computed as the perp distance between cloud points and the LS-fitted plane
            Format:    Scalar
            Type:    Boolean        
    output:
        grdpt_sigZ:    Terrain roughness defined by std dev of Z of the resampled dataset computed at the moving windows of wwid size
            Format:    scalar
            Type:    scalar
        grdpt_sigZ_dev:    Terrain roughness defined by std dev of deviation of Z of the resampled dataset computed at the moving windows of wwid size
            Format:    scalar
            Type:    scalar        
        grdpt_pts:    The number of points found within the moving window
            Format:    scalar
            Type:    scalar
    Important notes:
        If the number point situated within the moving window smaller than        'min_sample_no'  then 'grdpt_sigZ_dev' = NaN
        If the number point situated within the moving window smaller than max(3, 'min_sample_no') then     'grdpt_sigZ' = NaN
    '''
    absdX, absdY = abs(ptcloud[:,0] - grdpt[0]), abs(ptcloud[:,1] - grdpt[1]) # abs of deltaX and deltaY
    lstidx = [ii for ii in range(len(absdX)) if absdX[ii] <= wwid/2 and absdY[ii] <= wwid/2] # indices of all ptcloud located within the moving window
    grdpt_sigZ, grdpt_sigZ_dev = nan, nan # grdpt_sigZ = nan if len(lstidx) < min_sample_no; grdpt_sigZ_dev = nan if len(lstidx) < max(3, min_sample_no)
    if nondeviation and len(lstidx) >= max(1, min_sample_no): grdpt_sigZ = np.std(ptcloud[lstidx,2])
    if deviation    and len(lstidx) >= max(3, min_sample_no): # at least max of (3, min_sample_no points) points are needed to generated linear-fit plane then est deviations
        subptcloud = ptcloud[lstidx]
        COEF = cloud_plane_coef_bestfit_min_z(subptcloud) # coef of the eq. of a LS-fit plane
        subptcloud_dev = []
        for item in subptcloud: subptcloud_dev.append(pt_plane_dist(COEF, item))
        grdpt_sigZ_dev = np.std(np.array(subptcloud_dev))    
    grdpt_pts = len(lstidx)
    return (grdpt_sigZ, grdpt_sigZ_dev, grdpt_pts)

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

def cloud_plane_coef_bestfit_min_z(ptcloud): # Estimation of coefficients of the equation of a plane through a selection of points (X, Y, Z) by LS fit with minimisation of Z
    '''    The plane equation is: ax + by + z + d = 0 => -z = ax + by + d
    inputs:
        ptcloud:    Coordinates of the point cloud dataset
            Format: [npts, ncord] = [no of points, no of coordinates (3: [X, Y, Z])]
            Type:   array
    outputs:
        COEF:        LS-fit coefficients of the equation of a plane
            Format:    [ncoef] = [no of coefficients (4: a, b, c=1, d)]
            Type:    array
    '''
    COEF, _, _, _ = scipy.linalg.lstsq(np.hstack((ptcloud[:,:2], np.ones((ptcloud.shape[0], 1)))), -ptcloud[:,2])
    return np.array([COEF[0], COEF[1], 1, COEF[2]])

def pt_plane_dist(COEF, pt):
    '''    Estimation of the distance between a point to a plane described by coefficients of the plane eq.
        The plane eq is: ax + by + cz + d = 0
        The dist = (a*x_0 + b*y_0 + c*z_0 + d)/sqrt(a**2 + b**2 + c**2)
    inputs:
        COEF        Coefficients of the plane eq.
            Format:    [ncoef] = [no of coefficients (4: a, b, c, d)]
            Type:    array
        pt            coordinates of the considered point
            Format:    [ncord,] = [no of coordinates (3: [x, y, z])]
    outputs:
        dist:        Estimated distance
            Format:    scalar
            Type:    scalar
    '''
    return np.dot(COEF, np.hstack((pt,1)))/np.sqrt(sum(map(lambda x: x**2, COEF[:3])))

def prgrsTime(disstr, prgrs_prct, prgrs_runtime, barlen=50, end=False):
    prgrs_esttime = prgrs_runtime / prgrs_prct * (100 - prgrs_prct)    
    prgrs_len = int(prgrs_prct/100*barlen)
    eta_len   = barlen - prgrs_len
    prgrs_txt = disstr.expandtabs(8) + " |" + "="*prgrs_len + " "*eta_len + "| " + str(int(round(prgrs_runtime))) + " s (" + str(round(float(prgrs_runtime)/60, 1)) + " mins) / " + str(int(round(prgrs_esttime))) + " s (" + str(round(float(prgrs_esttime/60), 1)) + " mins)"    
    if end == False:
        print (prgrs_txt, end = "\r")
    else:
        print (prgrs_txt)

def cloud_dsamp_near_MP_starmap2Async(ptcloud, resgrdX, resgrdY, verbose=False, verbosetab='', verboseprct=1, nprocs=mp.cpu_count()): # Downsample a point cloud dataset by assigning the nearest point with multiprocessing starmapAsync
    ''' 
    input:
        ptcloud:    Coordinates of the point cloud dataset
            Format: [npts, ncord] = [no of points, no of coordinates (2: [X, Y] or 3: [X, Y, Z])]
            Type:   array
        resgrdX:    parameters of X grids
            Format:    [resmingrdX, resgrddX, resmaxgrdX]
            Type:    array
        resgrdX:    parameters of Y grids
            Format:    [resmingrdY, resgrddY, resmaxgrdY]
            Type:    array
    output:
        resptcloud:    Coordinates of cloud pts nearest to resampled grids
            Format:    [npts, ncord=2] = [no of points, no of coordinates (2: [X, Y])]
            Type:    array
        resptcloudidx:    Indices of cloud pts nearest to resampled grids
            Format:    [npts] = [no of points]
            Type:    list
    '''
    if verbose: print(verbosetab + 'Generate mesh grids based on resampled dX/dY: ' + str(resgrdX) + '/' + str(resgrdY))
    resmeshX, resmeshY = np.meshgrid(np.arange(resgrdX[0], resgrdX[2]+resgrdX[1], resgrdX[1]), np.arange(resgrdY[0], resgrdY[2]+resgrdY[1], resgrdY[1]))
    resptcloudgrd = np.array(np.column_stack((resmeshX.ravel(), resmeshY.ravel())))
    del resmeshX, resmeshY
    if verbose: print(verbosetab + 'Search cloud points that are nearest to mesh grids')
    _, resptcloudidx = cloud_nearest_search_many_MP_starmap2Async(resptcloudgrd, ptcloud, verbose=verbose, verbosetab=verbosetab+'\t', verboseprct=verboseprct, nprocs=nprocs)    
    del resptcloudgrd
    if verbose: print(verbosetab + 'Remove duplicated points, sort in ascending, then assign ptcloud at the nearest points to resptcloud')
    resptcloudidx = sorted(set(resptcloudidx))
    resptcloud    = np.array([ptcloud[idx,:] for idx in resptcloudidx])
    return resptcloud, resptcloudidx

def cloud_nearest_search_many_MP_starmap2Async(grdpt, ptcloud, verbose=False, verbosetab='', verboseprct=1, nprocs=mp.cpu_count()): # Call cloud_grid_nearest_search_many with multiprocessing.
    ''' input & output are the same as those in cloud_grid_nearest_search_many '''
    if nprocs < 2: raise ValueError('\r\n' + verbosetab + __name__ + '.' + inspect.currentframe().f_code.co_name + ": The number of workers 'nprocs' must be at least 2 \n")
    if nprocs > len(grdpt): raise ValueError('\r\n' + verbosetab + __name__ + '.' + inspect.currentframe().f_code.co_name + ": The number of workers 'nprocs' must be smaller than or equal to " + str(len(ptcloud)) + " \n")
    inp_lists = [[grdpt[item, :] for item in row] for row in slice_data_idx(len(grdpt), nprocs)]
    with mp.Pool(nprocs) as pool: results = pool.starmap_async(cloud_nearest_search_many_KDTree, [(item, ptcloud, verbose, verbosetab, verboseprct, ii+1) for ii, item in enumerate(inp_lists)]).get()    # this calls pool.starmap_async, normally faster than pool.starmap
    grdptdist, grdptidx = [item for sublist in results for item in sublist[0]], [item for sublist in results for item in sublist[1]]
    return grdptdist, grdptidx

def cloud_nearest_search_many_KDTree(grdpt, ptcloud, verbose=False, verbosetab='', verboseprct=1, worker=1): # Search pt in ptcloud that is nearest to pt in grdpt by scipy.spatial.KDTree
    '''
    input:
        grdpt:        Coordinates of grid points
            Format: [npts, ncord = 2 or 3] = [no of points, no of coordinates (2: [X, Y] or 3: [X, Y, Z])]
            Type:   array or list
        ptcloud:    Coordinates of the point cloud dataset
            Format: [npts, ncord = 2 or 3] = [no of points, no of coordinates (2: [X, Y] or 3: [X, Y, Z])]
            Type:   array
    output:
        grdptdist:    A list of distances to the nearest points
            Format:    [npts] = [no of points]
            Type:    List
        grdptidx:    A list of index of nearest points
            Format:    [npts] = [no of points]
            Type:    List '''
    tree = KDTree(ptcloud[:,:2])
    tic, intvlprct, grdptdist, grdptidx = time.time(), max(1, round(float(len(grdpt))*verboseprct/100)), [np.nan]*len(grdpt), [np.nan]*len(grdpt)
    for ii, item in enumerate(grdpt):        
        grdptdist[ii], grdptidx[ii] = tree.query(item, 1)
        if verbose:
            if (ii + 1) == 1 or (ii + 1) % intvlprct == 0 or (ii + 1) == len(grdpt):
                prgrs_prct, prgrs_runtime = float(ii + 1) / len(grdpt) * 100, time.time() - tic
                prgrs_str = verbosetab + 'Worker: ' + str(worker) + '; Estimated pt no: ' + str(ii + 1) + '/' + str(len(grdpt)) + ' [' + str(int(round(prgrs_prct))) + '%]'
                f = prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=True) if (ii + 1) == len(grdpt) else prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime, end=False)
    return grdptdist, grdptidx

def sigmZ_rho_Dlta_class(X_Var, Y_Var, X_step, X_tolr, min_samp_size=30):
    '''    Classification of interpolation error according to classes of terrain roughness or point density
        then estimate std dev of interploation error for each class
    Input:
        X_Var:    Terrain roughness or point density
            Format:    [npts, ] = [no of points, ]
        Y_Var:    Interpolation error
            Format:    [npts, ] = [no of points, ]
        X_step:    Step of classes of terrain roughness or point density
            Format:    Scalar
            Type:    Scalar
        X_tolr:    Tolerance of classes of terrain roughness or point density
            Format:    Scalar
            Type:    Scalar
        min_samp_size:    Minimum number of values over which std dev of Y_Var is estimated
            Format:    Scalar
            Type:    Scalar
    Outputs:
        X_Var_Class:    Classes of terrain roughness or point density 
            Format:    [ncls, ] = [no of class, ]
            Type:    Array
        Y_Var_Class:    Std dev of Y_Var corresponding to each class
            Format:    [ncls, ] = [no of class, ]
            Type:    Array
        Class_Val_No:    Number of values in each class
            Format:    [ncls, ] = [no of class, ]
            Type:    Array
    '''
    X_Var_class, Y_Var_class, Class_Val_No = np.arange(min(X_Var), max(X_Var), X_step), [], []
    for item in X_Var_class:
        lstidx = [jj for jj in range(len(X_Var)) if item - X_tolr <= X_Var[jj] <= item + X_tolr]
        Class_Val_No.append(len(lstidx))
        if len(lstidx) >= min_samp_size:
            Y_Var_class.append(np.std(Y_Var[lstidx]))
        else:
            Y_Var_class.append(np.nan)
    X_Var_class[0] = 0
    Y_Var_class,  Class_Val_No = np.array(Y_Var_class), np.array(Class_Val_No)
    return X_Var_class, Y_Var_class, Class_Val_No