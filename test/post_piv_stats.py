import statistics
import json
import numpy as np
import matplotlib.pyplot as  plt
from synthetic_dem_functions import create_dem, get_deformation, export_geotiff


def vector_residuals(vector_file, deform_params):
    with open(vector_file) as json_file:
        origins_vectors = np.asarray(json.load(json_file))

    # "Observed" displacements
    X = origins_vectors[:,0]
    Y = origins_vectors[:,1]
    vector_X = origins_vectors[:,2]
    vector_Y = origins_vectors[:,3]

    # "Computed" displacements
    dX, dY = get_deformation(X, Y, deform_params)

    # Residuals
    res_X = vector_X - dX
    res_Y = vector_Y - dY
    
    # Show some data
    std_X = statistics.stdev(res_X)
    std_Y = statistics.stdev(res_Y)
    print("std X = {}".format(std_X))
    print("std Y = {}".format(std_Y))

    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # axs[0].hist(res_X, bins=np.arange(-0.4, 0.425, 0.025))
    # axs[0].set_title('X std = {}'.format(std_X))
    # axs[0].set_xlim(-0.4, 0.4)
    # axs[0].set_ylim(0, 150)
    # axs[1].hist(res_Y, bins=np.arange(-.4, 0.425, 0.025))
    # axs[1].set_title('Y std = {}'.format(std_Y))
    # axs[1].set_xlim(-0.4, 0.4)
    # axs[1].set_ylim(0, 150)
    # plt.show()

    return res_X, res_Y


def uncertainty_components(covariance_file):
    with open(covariance_file) as json_file:
        endpoints_covariances = json.load(json_file)
    # print(endpoints_covariances)

    # Uncertainty in X and Y and semi-major axis of uncertainty ellipse
    std_X = []
    std_Y = []
    semimajor = []
    for ec in endpoints_covariances:
        cov = ec[1]
        std_X.append(np.sqrt(cov[0][0]))
        std_Y.append(np.sqrt(cov[1][1]))
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        semimajor.append(np.sqrt(2.298*np.amax(eigenvalues)))
    
    return std_X, std_Y, semimajor


def compare(res_X, res_Y, std_X, std_Y, semimajor):
    count = len(res_X)
    num_X = 0
    num_Y = 0
    num_mag = 0
    for (rx, ry, sx, sy, sm) in zip(res_X, res_Y, std_X, std_Y, semimajor):
        if abs(rx) <= sx:
            num_X += 1
        if abs(ry) <= sy:
            num_Y += 1
        if np.sqrt(rx**2 + ry**2) <= sm:
            num_mag += 1
    
    percent_X = (num_X/count) * 100
    percent_Y = (num_Y/count) * 100
    percent_mag = (num_mag/count) * 100

    print("Percent X = {}".format(percent_X))
    print("Percent Y = {}".format(percent_Y))
    print("Percent Magnitude = {}".format(percent_mag))

    return percent_X, percent_Y, percent_mag



vector_file = "tripleShear0p05_vectors.json"
covariance_file = "tripleShear0p05_covariances.json"
deform_params = {
        'tx': 0,
        'ty': 0,
        'sx': 0,
        'sy': 0.05,
        'g_maj': 30,
        'g_min': 30,
        'g_amp': 0,
        'g_az': 0
    }

res_X, res_Y = vector_residuals(vector_file, deform_params)
std_X, std_Y, semimajor = uncertainty_components(covariance_file)
percent_X, percent_Y, percent_mag = compare(res_X, res_Y, std_X, std_Y, semimajor)
