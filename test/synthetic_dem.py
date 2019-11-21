"""
Synthetic DEM surface creation using overlapping 2D Gaussians.
"""
import numpy as  np
import matplotlib.pyplot as plt


def create_dem(dem_size, num_gsn, gsn_max, tx, ty, gxstd, gystd):

    half_data_size = np.floor(dem_size*1.2/2)

    # Uniform random distributions of the gaussian magnitudes and widths
    np.random.seed(1)
    mag = -gsn_max/2 + gsn_max*np.random.rand(num_gsn, 1)
    wid = mag

    # Uniform random distribution of the gaussian locations
    x_gsn = -half_data_size + 2*half_data_size*np.random.rand(num_gsn, 1)
    y_gsn = -half_data_size + 2*half_data_size*np.random.rand(num_gsn, 1)

    # X, Y coords of synthetic DEM
    half_dem_size = np.floor(dem_size/2)
    x_vec = np.arange(-half_dem_size, half_dem_size, 1)
    y_vec = np.arange(-half_dem_size, half_dem_size, 1)
    X, Y = np.meshgrid(x_vec, y_vec)
    
    # create DEM surface elevations from random gaussians
    Z = np.zeros(X.shape)
    for i in range(num_gsn):
        Z += mag[i]*np.exp(-((((X-x_gsn[i])**2) / (2*wid[i]**2))
                             +(((Y-y_gsn[i])**2) / (2*wid[i]**2))))
    
    plt.imshow(Z)
    plt.show()




