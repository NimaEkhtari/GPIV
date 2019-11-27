import sys
import json
import math

import numpy as np
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle


def show(image_file, vector_file="", covariance_file="",
         vector_scale=1, ellipse_scale=1):
    # Clean input
    if vector_scale is None:
        vector_scale = 1
    if ellipse_scale is None:
        ellipse_scale = 1
    
    # Get background image and plot
    (image, geo_extents, geo_transform) = get_image_array(image_file)
    figure, axes = plot_image(image, geo_extents)

    # Overlay PIV vectors and uncertainty ellipses if requested
    if vector_file:
        with open(vector_file) as json_file:
            origins_vectors = json.load(json_file)
        plot_vectors(axes, geo_extents,
                     origins_vectors, vector_scale)
    if covariance_file:
        with open(covariance_file) as json_file:
            endpoints_covariances = json.load(json_file)
        plot_ellipses(axes, geo_extents,
                      endpoints_covariances, ellipse_scale)

    plt.show()


def get_image_array(image_file):
    image_source = rasterio.open(image_file)
    image_array = image_source.read(1)
    geo_transform = np.reshape(np.asarray(image_source.transform), (3,3))
    # Geo extent order is [left, right, bottom, top]
    geo_extents = list(rasterio.plot.plotting_extent(image_source)) 
    image_source.close()
    return image_array, geo_extents, geo_transform


def plot_image(image_array, geo_extents):
    figure = plt.figure(figsize=(6,6))
    axes = plt.gca()
    image_data_min = min(np.percentile(image_array, 1),
                         np.percentile(image_array, 1))
    image_data_max = max(np.percentile(image_array, 99),
                         np.percentile(image_array, 99))
    plt.imshow(image_array,
               cmap=plt.cm.gray,
               extent=geo_extents,
               vmin=image_data_min,
               vmax=image_data_max)
    return figure, axes


def plot_vectors(axes, geo_extents, origins_vectors, vector_scale_factor):
    plot_width_in_pixels = axes.get_window_extent().width
    plot_width_in_ground_units = geo_extents[1] - geo_extents[0]
    pixels_per_ground_unit = plot_width_in_pixels / plot_width_in_ground_units
    ground_units_per_pixel = plot_width_in_ground_units / plot_width_in_pixels

    origins_vectors = np.asarray(origins_vectors)
    vector_lengths_ground = np.linalg.norm(origins_vectors[:,2:], axis=1)
    vector_lengths_pixels = vector_lengths_ground * pixels_per_ground_unit

    arrow_scale_factor = ((30*ground_units_per_pixel)
                           / np.median(vector_lengths_ground))
    arrow_head_scale_factor = 8*ground_units_per_pixel

    origins_vectors = np.asarray(origins_vectors)
    for i in range(len(origins_vectors)):
        arrow = FancyArrow(
            origins_vectors[i][0],
            origins_vectors[i][1],
            origins_vectors[i][2] * arrow_scale_factor * vector_scale_factor,
            # Assumes vectors components are defined with up as positive
            origins_vectors[i][3] * arrow_scale_factor * vector_scale_factor,
            length_includes_head=True,
            head_width=arrow_head_scale_factor,
            overhang=0.8,
            fc='yellow',
            ec = 'yellow'
        )
        axes.add_artist(arrow)

    geo_height = geo_extents[3] - geo_extents[2]
    legend_background = Rectangle(
        (geo_extents[0] + geo_height/50, geo_extents[2] + geo_height/50),
        geo_height/7,
        geo_height/7,
        fc='silver',
        clip_on=False,
        alpha=0.75
    )
    axes.add_artist(legend_background)
    plt.text(
        geo_extents[0] + geo_height/50 + geo_height/14,
        geo_extents[2] + geo_height/7,
        '{0:.3f}'.format(np.median(vector_lengths_ground)/vector_scale_factor),
        horizontalalignment='center',
        verticalalignment='top'
    )
    arrow = FancyArrow(
        geo_extents[0] + geo_height/50
            + (geo_height/7 - 30*ground_units_per_pixel) / 2,
        geo_extents[2] + geo_height/14,
        30*ground_units_per_pixel,
        0,
        length_includes_head=True,
        head_width=arrow_head_scale_factor,
        overhang=0.8,
        fc='yellow',
        ec = 'yellow'
    )
    axes.add_artist(arrow)


def plot_ellipses(axes, geo_extents, endpoints_covariances, user_scale_factor):
    plot_width_in_pixels = axes.get_window_extent().width
    plot_width_in_ground_units = geo_extents[1] - geo_extents[0]
    pixels_per_ground_unit = plot_width_in_pixels / plot_width_in_ground_units
    ground_units_per_pixel = plot_width_in_ground_units / plot_width_in_pixels

    semimajor_lengths_ground = []
    semimajor_lengths_pixels = []
    for i in range(len(endpoints_covariances)):
        eigenvalues, eigenvectors = np.linalg.eig(endpoints_covariances[i][1])
        max_index = np.argmax(eigenvalues)
        semimajor_lengths_ground.append(math.sqrt(2.298*eigenvalues[max_index]))
    ellipse_scale_factor = (
        (20*ground_units_per_pixel) / np.median(semimajor_lengths_ground)
    )

    for i in range(len(endpoints_covariances)):
        eigenvalues, eigenvectors = np.linalg.eig(endpoints_covariances[i][1])
        max_index = np.argmax(eigenvalues)
        min_index = np.argmin(eigenvalues)
        semimajor = math.sqrt(2.298*eigenvalues[max_index])
        semiminor = math.sqrt(2.298*eigenvalues[min_index])
        my_angle = np.degrees(
            np.arctan(eigenvectors[max_index][1]/eigenvectors[max_index][0])
        )
        ellipse = Ellipse(
            (endpoints_covariances[i][0][0], endpoints_covariances[i][0][1]),
            semimajor * ellipse_scale_factor * user_scale_factor,
            semiminor * ellipse_scale_factor * user_scale_factor,
            angle=my_angle,
            fc='None',
            ec='red'
        )
        axes.add_artist(ellipse)

    geo_height = geo_extents[3] - geo_extents[2]
    legend_background = Rectangle(
        (geo_extents[0] + geo_height/50 + geo_height/7 + geo_height/50,
        geo_extents[2] + geo_height/50),
        geo_height/7,
        geo_height/7,
        fc='silver',
        clip_on=False,
        alpha=0.75
    )
    axes.add_artist(legend_background)
    plt.text(geo_extents[0]
                + geo_height/50
                + geo_height/7
                + geo_height/50
                + geo_height/14,
             geo_extents[2]
                + geo_height/7,
             '{0:.3f}'.format(np.median(semimajor_lengths_ground)
                / user_scale_factor),
             horizontalalignment='center',
             verticalalignment='top')
    ell = Ellipse((geo_extents[0] + geo_height/50
                                  + geo_height/7
                                  + geo_height/50
                                  + geo_height/14,
                   geo_extents[2] + geo_height/14),
                   20*ground_units_per_pixel,
                   20*ground_units_per_pixel, 
                   ec='red',
                   fc='none',
                   clip_on=False)
    axes.add_artist(ell)