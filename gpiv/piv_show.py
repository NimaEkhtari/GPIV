import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import FancyArrow
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
import json


def show(image_file, vector_file=None, ellipse_file=None, scale_factor=None):

    (image_array,image_geo_extents, image_geo_transform) = get_image_array(image_file)
    (figure, axes) = plot_image(image_array, image_geo_extents)

    if scale_factor is None:
        scale_factor = 1
    if ellipse_file is not None:
        plot_ellipses(axes, image_geo_extents, ellipse_file, float(scale_factor))
    if vector_file is not None:
        plot_vectors(axes, image_geo_extents, vector_file, float(scale_factor))
    
    plt.show()


def get_image_array(image_file):

    image_source = rasterio.open(image_file)
    image_array = image_source.read(1)
    image_geo_transform = np.reshape(np.asarray(image_source.transform), (3,3))
    image_geo_extents = list(rasterio.plot.plotting_extent(image_source)) # [left, right, bottom, top]
    image_source.close()

    return image_array, image_geo_extents, image_geo_transform


def plot_image(image_array, image_geo_extents):

    figure = plt.figure(figsize=(6,6))
    axes = plt.gca()

    image_data_min = min(np.percentile(image_array, 1), np.percentile(image_array, 1))
    image_data_max = max(np.percentile(image_array, 99), np.percentile(image_array, 99))
    plt.imshow(image_array,
               cmap=plt.cm.gray,
               extent=image_geo_extents,
               vmin=image_data_min,
               vmax=image_data_max)

    return figure, axes


def plot_vectors(axes, image_geo_extents, vector_file, user_scale_factor):

    with open(vector_file) as json_file:
        origins_vectors = json.load(json_file)
    origins_vectors_numpy = np.asarray(origins_vectors)

    plot_width_in_pixels = axes.get_window_extent().width
    plot_width_in_ground_units = image_geo_extents[1] - image_geo_extents[0]
    pixels_per_ground_unit = plot_width_in_pixels / plot_width_in_ground_units
    ground_units_per_pixel = plot_width_in_ground_units / plot_width_in_pixels

    vector_lengths_ground = np.linalg.norm(origins_vectors_numpy[:,2:], axis=1)
    vector_lengths_pixels = vector_lengths_ground * pixels_per_ground_unit

    arrow_scale_factor = (30*ground_units_per_pixel) / np.median(vector_lengths_ground)
    arrow_head_scale_factor = 10*ground_units_per_pixel

    for i in range(len(origins_vectors)):
        arrow = FancyArrow(
            origins_vectors[i][0],
            origins_vectors[i][1],
            origins_vectors[i][2] * arrow_scale_factor * user_scale_factor,
            -origins_vectors[i][3] * arrow_scale_factor * user_scale_factor,  # Negative sign converts from dV (positive down) to dY (positive up)
            length_includes_head=True,
            head_width=arrow_head_scale_factor,
            overhang=0.8,
            fc='yellow',
            ec = 'yellow'
        )
        axes.add_artist(arrow)
    
    geo_height = image_geo_extents[3] - image_geo_extents[2]
    legend_background = Rectangle((image_geo_extents[0] + geo_height/50, image_geo_extents[2] + geo_height/50),
                        geo_height/7, geo_height/7, 
                        fc='silver', clip_on=False, alpha=0.5)    
    axes.add_artist(legend_background)
    plt.text(image_geo_extents[0] + geo_height/50 + geo_height/14,
             image_geo_extents[2] + geo_height/7,
             '{0:.3f}'.format(np.median(vector_lengths_ground)),
             horizontalalignment='center', verticalalignment='top')
    arrow = FancyArrow(
        image_geo_extents[0] + geo_height/50 + (geo_height/7 - 30*ground_units_per_pixel)/2,
        image_geo_extents[2] + geo_height/14,
        30*ground_units_per_pixel, 0,
        length_includes_head=True, head_width=arrow_head_scale_factor,
        overhang=0.8, fc='yellow', ec = 'yellow')
    axes.add_artist(arrow)


def plot_ellipses(axes, image_geo_extents, ellipse_file, user_scale_factor):

    with open(ellipse_file) as json_file:
        locations_covariances = json.load(json_file)
    
    plot_width_in_pixels = axes.get_window_extent().width
    plot_width_in_ground_units = image_geo_extents[1] - image_geo_extents[0]
    pixels_per_ground_unit = plot_width_in_pixels / plot_width_in_ground_units
    ground_units_per_pixel = plot_width_in_ground_units / plot_width_in_pixels

    semimajor_lengths_ground = []
    semimajor_lengths_pixels = []
    for i in range(len(locations_covariances)):
        eigenvalues, eigenvectors = np.linalg.eig(locations_covariances[i][1])
        max_index = np.argmax(eigenvalues)
        semimajor_lengths_ground.append(math.sqrt(2.298*eigenvalues[max_index]))
    ellipse_scale_factor = (20*ground_units_per_pixel) / np.median(semimajor_lengths_ground)

    for i in range(len(locations_covariances)):
        eigenvalues, eigenvectors = np.linalg.eig(locations_covariances[i][1])
        max_index = np.argmax(eigenvalues)
        min_index = np.argmin(eigenvalues)
        semimajor = math.sqrt(2.298*eigenvalues[max_index])
        semiminor = math.sqrt(2.298*eigenvalues[min_index])
        my_angle = np.degrees(np.arctan(eigenvectors[max_index][1]/eigenvectors[max_index][0]))
        ellipse = Ellipse(
            (locations_covariances[i][0][0], locations_covariances[i][0][1]),
            semimajor * ellipse_scale_factor * user_scale_factor,
            semiminor * ellipse_scale_factor * user_scale_factor,
            angle=my_angle,
            fc='None',
            ec='red'
        )
        axes.add_artist(ellipse)

    geo_height = image_geo_extents[3] - image_geo_extents[2]
    legend_background = Rectangle((image_geo_extents[0] + geo_height/50 + geo_height/7 + geo_height/50,
                                   image_geo_extents[2] + geo_height/50),
                                   geo_height/7, geo_height/7, fc='silver', clip_on=False, alpha=0.5)
    axes.add_artist(legend_background)
    plt.text(image_geo_extents[0] + geo_height/50 + geo_height/7 + geo_height/50 + geo_height/14,
             image_geo_extents[2] + geo_height/7,
             '{0:.3f}'.format(np.median(semimajor_lengths_ground)),
             horizontalalignment='center', verticalalignment='top')
    ell = Ellipse((image_geo_extents[0] + geo_height/50 + geo_height/7 + geo_height/50 + geo_height/14, image_geo_extents[2] + geo_height/14),
                   20*ground_units_per_pixel, 20*ground_units_per_pixel, 
                   ec='red', fc='none', clip_on=False)
    axes.add_artist(ell)