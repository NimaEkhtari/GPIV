import statistics
import json
import numpy as np
import matplotlib.pyplot as  plt
from synthetic_dem_functions import create_dem, get_deformation, export_geotiff


def compare_to_known(vector_file, params):
    with open(vector_file) as json_file:
        origins_vectors = json.load(json_file)
    origins_vectors = np.asarray(origins_vectors)
    X = origins_vectors[:,0]
    Y = origins_vectors[:,1]
    vX = origins_vectors[:,2]
    vY = origins_vectors[:,3]

    dX, dY = get_deformation(X, Y, deform_params)

    error_u = vX-dX
    error_v = vY-dY
    std_u = statistics.stdev(error_u)
    std_v = statistics.stdev(error_v)

    print("std du = {}".format(std_u))
    print("std dv = {}".format(std_v))

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(error_u, bins=np.arange(-0.4, 0.425, 0.025))
    axs[0].set_title('u std = {}'.format(std_u))
    axs[0].set_xlim(-0.4, 0.4)
    axs[0].set_ylim(0, 150)
    axs[1].hist(error_v, bins=np.arange(-.4, 0.425, 0.025))
    axs[1].set_title('v std = {}'.format(std_v))
    axs[1].set_xlim(-0.4, 0.4)
    axs[1].set_ylim(0, 150)
    plt.show()


vector_file = "test/triple2_vectors.json"
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

compare_to_known(vector_file, deform_params)
