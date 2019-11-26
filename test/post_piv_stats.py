import numpy as np
from synthetic_dem import get_deformation
import statistics


def compare_to_known(vector_file, params):
    with open(vector_file) as json_file:
        origins_vectors = json.load(json_file)
    
    deform_params = {
        'tx': 0,
        'ty': 0,
        'sx': 0.5,
        'sy': 0,
        'g_maj': 1,
        'g_min': 1,
        'g_amp': 0,
        'g_az': 0
    }

    dX, dY = get_deformation(origins_vectors[:,0], origins_vectors[:,1], deform_params)

    du = [abs(u-dx) for u in origins_vectors[:,2] for dx in dX]
    dv = [abs(v-(-dy)) for v in origins_vectors[:,3] for dy in dY]

    print("std u = {}".format(statistics.stdev(du)))
    print("std v = {}".format(statistics.stdev(dv)))

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(du)
    axs[1].hist(dv)
    plt.show()



vector_file = "vectors.json"
covariance_file = "covariance.json"