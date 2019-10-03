import rasterio
import numpy as np
import sys
from skimage.feature import match_template
import matplotlib.pyplot as plt
import matplotlib.patches
import math
import json
import show_functions
from scipy import interpolate


def piv(before_height_file, after_height_file,
        template_size, step_size,
        before_uncertainty_file, after_uncertainty_file,
        propagate, output_base_name):

    (before_height, before_uncertainty, 
     after_height, after_uncertainty, 
     geo_transform) = get_image_arrays(before_height_file,
                                       before_uncertainty_file,
                                       after_height_file,
                                       after_uncertainty_file,
                                       propagate)

    piv_iterator = PivIterator(before_height, before_uncertainty,
                              after_height, after_uncertainty)

    if not propagate:
        print("Computing PIV.")
        piv_iterator.iterate(propagate, template_size*4, step_size*4)



        # run_piv(before_height, [],
        #         after_height, [],
        #         geo_transform, template_size, step_size,
        #         False, output_base_name)
        # show_functions.show(before_height_file,
        #                     output_base_name + 'vectors.json',
        #                     None,
        #                     1, 1)
    # else:
    #     print("Computing bias variance.")
    #     run_piv(before_height, [],
    #             before_height, [],
    #             geo_transform, template_size, step_size,
    #             False, output_base_name)
    #     xy_bias_variance = get_bias_variance(output_base_name)

    #     print("Computing PIV and propagating uncertainty.")
    #     run_piv(before_height, before_uncertainty,
    #             after_height, after_uncertainty,
    #             geo_transform, template_size, step_size,
    #             True, output_base_name)

    #     print("Adding bias variance to propagated PIV uncertainty.")
    #     add_bias_variance(output_base_name, xy_bias_variance)

    #     show_functions.show(before_height_file,
    #                         output_base_name + 'vectors.json',
    #                         output_base_name + 'covariances.json',
    #                         1, 1)


class PivIterator:
    def __init__(self, 
                 before_height, before_uncertainty,
                 after_height, after_uncertainty):
        self._before_height = before_height
        self._before_uncertainty = before_uncertainty
        self._after_height = after_height
        self._after_height_deformed = after_height
        self._after_uncertainty = after_uncertainty
        self._after_uncertainty_deformed = after_uncertainty
        self._deformation_field = None
        self._deformation_field_total = None


    def iterate(self, propagate, template_size, step_size):
        piv_origins = []
        piv_vectors = []
        if propagate:
            peak_covariance = []
            numeric_partial_derivative_increment = 0.000001

        status_figure = plt.figure()
        before_axis = plt.subplot(1, 2, 1)
        after_axis = plt.subplot(1, 2, 2)

        search_size = template_size * 2 # size of area to be searched for match in 'after' image
        number_horizontal_computations = math.floor((self._before_height.shape[1] - search_size) / step_size)
        number_vertical_computations = math.floor((self._before_height.shape[0] - search_size) / step_size)

        for vt_count in range(number_vertical_computations):
            for hz_count in range(number_horizontal_computations):

                hz_template_start = int(hz_count*step_size + math.ceil(template_size/2))
                hz_template_end = int(hz_count*step_size + math.ceil(template_size/2) + template_size)
                vt_template_start = int(vt_count*step_size + math.ceil(template_size/2))
                vt_template_end = int(vt_count*step_size + math.ceil(template_size/2) + template_size)
                height_template = self._before_height[vt_template_start:vt_template_end, hz_template_start:hz_template_end].copy()
                
                hz_search_start = int(hz_count*step_size)
                hz_search_end = int(hz_count*step_size + search_size + (template_size % 2)) # the modulo addition forces the search area to be symmetric around odd-sized templates
                vt_search_start = int(vt_count*step_size)
                vt_search_end = int(vt_count*step_size + search_size + (template_size % 2)) 
                height_search = self._after_height[vt_search_start:vt_search_end, hz_search_start:hz_search_end].copy()            

                self._show_piv_location(before_axis, after_axis,
                                        hz_template_start, vt_template_start,
                                        hz_search_start, vt_search_start,
                                        template_size, search_size)     

                # guard against flat areas, which produce a divide by zero in the correlation
                # guard agains NaN values, which breaks scipy's match_template function
                if (np.max(height_template)-np.min(height_template) < 1e-10 or
                        np.max(height_search)-np.min(height_search) < 1e-10 or
                        np.isnan(height_template).any() or
                        np.isnan(height_search).any()):
                    continue

                normalized_cross_correlation = match_template(height_search, height_template) # uses FFT based correlation
                correlation_max = np.where(normalized_cross_correlation == np.amax(normalized_cross_correlation))

                # peak location on edges of correlation matrix breaks sub-pixel peak interpolation
                if (correlation_max[0][0]==0 or
                        correlation_max[1][0]==0 or 
                        correlation_max[0][0]==normalized_cross_correlation.shape[0]-1 or 
                        correlation_max[1][0]==normalized_cross_correlation.shape[1]-1): 
                    continue

                subpixel_peak = self._get_subpixel_peak(normalized_cross_correlation[
                    correlation_max[0][0]-1:correlation_max[0][0]+2,
                    correlation_max[1][0]-1:correlation_max[1][0]+2])
                
                piv_origins.append(((hz_count*step_size + template_size - (1 - template_size % 2)*0.5), # modulo operator adjusts even-sized template origins to be between pixel centers
                                    (vt_count*step_size + template_size - (1 - template_size % 2)*0.5)))
                piv_vectors.append(((correlation_max[1][0] - math.ceil(template_size/2) + subpixel_peak[0]),
                                    (correlation_max[0][0] - math.ceil(template_size/2) + subpixel_peak[1])))

                if propagate:
                    uncertainty_template = self._before_uncertainty[vt_template_start:vt_template_end, hz_template_start:hz_template_end].copy()
                    uncertainty_search = self._after_uncertainty[vt_search_start:vt_search_end, hz_search_start:hz_search_end].copy()    

                    # propagate raster error into the 3x3 patch of correlation values that are centered on the correlation peak
                    correlation_covariance = self._propagate_pixel_into_correlation(
                        height_template,
                        uncertainty_template, 
                        height_search[correlation_max[0][0]-1:correlation_max[0][0]+template_size+1, correlation_max[1][0]-1:correlation_max[1][0]+template_size+1], # templateSize+2 x templateSize+2 subarray of the search array,
                        uncertainty_search[correlation_max[0][0]-1:correlation_max[0][0]+template_size+1, correlation_max[1][0]-1:correlation_max[1][0]+template_size+1], # templateSize+2 x templateSize+2 subarray of the search error array
                        normalized_cross_correlation[correlation_max[0][0]-1:correlation_max[0][0]+2, correlation_max[1][0]-1:correlation_max[1][0]+2], # 3x3 array of correlation values centered on the correlation peak
                        numeric_partial_derivative_increment) 

                    # propagate the correlation covariance into the subpixel peak location
                    subpixel_peak_covariance = self._propagate_correlation_into_subpixel_peak(
                        normalized_cross_correlation[correlation_max[0][0]-1:correlation_max[0][0]+2, correlation_max[1][0]-1:correlation_max[1][0]+2],
                        correlation_covariance,
                        subpixel_peak,
                        numeric_partial_derivative_increment)
                    
                    peak_covariance.append(subpixel_peak_covariance.tolist())   

        plt.close(status_figure)

        # bilinear interpolation of grid of vectors for each pixel
        # then use the interpolated vectors to deform 'after' images using cubic spline interpolation
        piv_origins = np.asarray(piv_origins)
        piv_vectors = np.asarray(piv_vectors)
        image_u_coords = np.arange(self._after_height.shape[1])
        image_v_coords = np.arange(self._after_height.shape[0])
        bilinear_u = interpolate.interp2d(piv_origins[:,0], piv_origins[:,1], piv_vectors[:,0])
        bilinear_v = interpolate.interp2d(piv_origins[:,0], piv_origins[:,1], piv_vectors[:,1])
        image_u_vector_component = bilinear_u(image_u_coords, image_v_coords)
        image_v_vector_component = bilinear_v(image_u_coords, image_v_coords)
        plt.figure(figsize=(10,10))
        plt.quiver(image_u_coords,image_v_coords,image_u_vector_component,-image_v_vector_component,angles='xy',scale_units='xy')
        plt.axis('equal')
        plt.show()

    def _show_piv_location(self,
                           before_axis, after_axis,
                           hz_template_start, vt_template_start,
                           hz_search_start, vt_search_start,
                           template_size, search_size):
        plt.sca(before_axis)
        plt.cla()
        before_axis.set_title('Before')
        before_axis.imshow(self._before_height, cmap=plt.cm.gray)
        before_axis.add_patch(matplotlib.patches.Rectangle(
            (hz_template_start, vt_template_start), 
            template_size-1, 
            template_size-1, 
            linewidth=1, 
            edgecolor='r',
            fill=None))
        
        plt.sca(after_axis)
        plt.cla()
        after_axis.set_title('After')            
        after_axis.imshow(self._after_height_deformed, cmap=plt.cm.gray)            
        after_axis.add_patch(matplotlib.patches.Rectangle(
            (hz_search_start,vt_search_start), 
            search_size-1, 
            search_size-1, 
            linewidth=1, 
            edgecolor='r',
            fill=None))

        plt.pause(0.1)


    def _get_subpixel_peak(self, normalized_cross_correlation):
        dx = (normalized_cross_correlation[1,2] - normalized_cross_correlation[1,0]) / 2
        dxx = normalized_cross_correlation[1,2] + normalized_cross_correlation[1,0] - 2*normalized_cross_correlation[1,1]
        dy = (normalized_cross_correlation[2,1] - normalized_cross_correlation[0,1]) / 2
        dyy = normalized_cross_correlation[2,1] + normalized_cross_correlation[0,1] - 2*normalized_cross_correlation[1,1]
        dxy = (normalized_cross_correlation[2,2] - normalized_cross_correlation[2,0] - normalized_cross_correlation[0,2] + normalized_cross_correlation[0,0]) / 4
        
        # hz_delta is postive left-to-right; vt_delta is postive top-to-bottom
        hz_delta = -(dyy*dx - dxy*dy) / (dxx*dyy - dxy*dxy)
        vt_delta = -(dxx*dy - dxy*dx) / (dxx*dyy - dxy*dxy)

        return [hz_delta, vt_delta]


    def _propagate_pixel_into_correlation(self,
                                          height_template,
                                          uncertainty_template, 
                                          height_search,
                                          uncertainty_search,
                                          normalized_cross_correlation,
                                          numeric_partial_derivative_increment):
        template_covariance_vector = np.square(uncertainty_template.reshape(uncertainty_template.size,)) # convert array to vector, row-by-row, and square the standard deviations into variances
        search_covariance_vector = np.square(uncertainty_search.reshape(uncertainty_search.size,))
        covariance_matrix = np.diag(np.hstack((template_covariance_vector, search_covariance_vector)))

        jacobian = self._get_correlation_jacobian(height_template, height_search, normalized_cross_correlation, numeric_partial_derivative_increment)    
        # Propagate the template and search area errors into the 9 correlation elements
        # The covariance order is by row of the normalized_cross_correlation (ncc) array (i.e., ncc[0,0], ncc[0,1], ncc[0,2], ncc[1,0], ncc[1,1], ...)
        correlation_covariance = np.matmul(jacobian,np.matmul(covariance_matrix,jacobian.T))

        return correlation_covariance


    def _get_correlation_jacobian(self, template, search,
                                  normalized_cross_correlation,
                                  numeric_partial_derivative_increment):

        number_template_rows, number_template_columns = template.shape
        number_search_rows, number_search_columns = search.shape
        jacobian = np.zeros((9, template.size + search.size))
        normalized_template = (template - np.mean(template)) / (np.std(template))

        # cycle through the 3x3 correlation array
        for row_correlation in range(3):
            for col_correlation in range(3):
                search_subarea = search[row_correlation:row_correlation+number_template_rows, col_correlation:col_correlation+number_template_columns]
                normalized_search_subarea = (search_subarea - np.mean(search_subarea)) / (np.std(search_subarea))

                template_partial_derivatives = np.zeros((number_template_rows, number_template_columns))
                search_partial_derivatives = np.zeros((number_search_rows, number_search_columns))

                # cycle through each pixel in the template and the search subarea and numerically estimate
                # its partial derivate with respect to the normalized cross correlation
                for row_template in range(number_template_rows):
                    for col_template in range(number_template_columns):
                        perturbed_template = template.copy()
                        perturbed_template[row_template,col_template] += numeric_partial_derivative_increment
                        perturbed_search_subarea = search_subarea.copy()
                        perturbed_search_subarea[row_template,col_template] += numeric_partial_derivative_increment

                        # spatial domain normalized cross correlation (i.e., does not use the FFT)
                        # about 20x faster than using skimage's FFT based match_template method
                        normalized_perturbed_template = (perturbed_template - np.mean(perturbed_template)) / (np.std(perturbed_template))
                        normalized_perturbed_search_subarea = (perturbed_search_subarea - np.mean(perturbed_search_subarea)) / (np.std(perturbed_search_subarea))
                        perturbed_template_normalized_cross_correlation = np.sum(normalized_perturbed_template * normalized_search_subarea) / template.size
                        perturbed_search_subarea_normalized_cross_correlation = np.sum(normalized_template * normalized_perturbed_search_subarea) / template.size
                        
                        # storage location adjustment by row_correlation and col_correlation accounts for the larger size of the search area than the template area
                        template_partial_derivatives[row_template, col_template] = (perturbed_template_normalized_cross_correlation - normalized_cross_correlation[row_correlation,col_correlation]) / numeric_partial_derivative_increment
                        search_partial_derivatives[row_correlation+row_template, col_correlation+col_template] = (perturbed_search_subarea_normalized_cross_correlation - normalized_cross_correlation[row_correlation, col_correlation]) / numeric_partial_derivative_increment 

                # reshape the partial derivatives from their current array form to vector form and store in the Jacobian
                # we match the row-by-row pattern used to form the covariance matrix in the calling function
                jacobian[row_correlation*3+col_correlation, 0:template.size] = template_partial_derivatives.reshape(template_partial_derivatives.size,)
                jacobian[row_correlation*3+col_correlation, template.size:template.size+search.size] = search_partial_derivatives.reshape(search_partial_derivatives.size,)

        return jacobian


    def _propagate_correlation_into_subpixel_peak(correlation,
                                                  correlation_covariance,
                                                  subpixel_peak,
                                                  numeric_partial_derivative_increment):
        jacobian = np.zeros((2,9))
        # cycle through the 3x3 correlation array, row-by-row, and create the jacobian matrix
        for row_correlation in range(3):
            for col_correlation in range(3):
                perturbed_correlation = correlation.copy()
                perturbed_correlation[row_correlation,col_correlation] += numeric_partial_derivative_increment            
                perturbed_hz_delta, perturbed_vt_delta = get_subpixel_peak(perturbed_correlation)            
                jacobian[0,row_correlation*3+col_correlation] = (perturbed_hz_delta - subpixel_peak[0]) / numeric_partial_derivative_increment
                jacobian[1,row_correlation*3+col_correlation] = (perturbed_vt_delta - subpixel_peak[1]) / numeric_partial_derivative_increment        
        # propagate the 3x3 array of correlation uncertainties into the sub-pixel U and V direction offsets
        subpixel_peak_covariance = np.matmul(jacobian, np.matmul(correlation_covariance, jacobian.T))
            
        return subpixel_peak_covariance


    # def deform(self):


    # def export(self, file_prefix):


    # def compute_bias(self, template_size, step_size):


# def run_piv(before_height, before_uncertainty,
#             after_height, after_uncertainty,
#             geo_transform, template_size,
#             step_size, propagate, output_base_name):

#     export_piv(piv_origins, piv_vectors,
#                geo_transform, output_base_name)

#     if propagate:
#         export_uncertainty(piv_origins, piv_vectors,
#                            peak_covariance, geo_transform, 
#                            output_base_name)


def get_image_arrays(
    before_height_file,
    before_uncertainty_file,
    after_height_file,
    after_uncertainty_file,
    propagate):    

    before_height_source = rasterio.open(before_height_file)
    before_height = before_height_source.read(1)
    after_height_source = rasterio.open(after_height_file)
    after_height = after_height_source.read(1)

    if propagate:
        before_uncertainty = rasterio.open(before_uncertainty_file).read(1)
        after_uncertainty = rasterio.open(after_uncertainty_file).read(1)
    else:
        before_uncertainty = []
        after_uncertainty = []

    # get raster coordinate transformation for later use
    before_geo_transform = np.reshape(np.asarray(before_height_source.transform), (3,3))
    after_geo_transform = np.reshape(np.asarray(after_height_source.transform), (3,3))
    if not np.array_equal(before_geo_transform, after_geo_transform):
        print("The extent and/or datum of the 'before' and 'after' DEMs is not equivalent.")
        sys.exit()    

    return before_height, before_uncertainty, after_height, after_uncertainty, before_geo_transform


def export_piv(
    piv_origins,
    piv_vectors,
    geo_transform,
    output_base_name):

    # Convert from pixels to ground distance
    piv_origins = np.asarray(piv_origins)        
    piv_origins *= geo_transform[0,0]  # Scale by pixel ground size
    piv_origins[:,0] += geo_transform[0,2]  # Offset by leftmost pixel to get ground coordinate
    piv_origins[:,1] = geo_transform[1,2] - piv_origins[:,1]  # Subtract from uppermost pixel to get ground coordinate    
    piv_vectors = np.asarray(piv_vectors)
    piv_vectors *= geo_transform[0,0]  # Scale by pixel ground size
    
    origins_vectors = np.concatenate((piv_origins, piv_vectors), axis=1)
    json.dump(origins_vectors.tolist(), open(output_base_name + "vectors.json", "w"))
    print("PIV displacement vectors saved to file '{}vectors.json'".format(output_base_name))


def export_uncertainty(piv_origins,
                       piv_vectors,
                       peak_covariance,
                       geo_transform,
                       output_base_name):

    # Convert from pixels to ground distance
    piv_origins = np.asarray(piv_origins)        
    piv_origins *= geo_transform[0,0]  # Scale by pixel ground size
    piv_origins[:,0] += geo_transform[0,2]  # Offset by leftmost pixel to get ground coordinate
    piv_origins[:,1] = geo_transform[1,2] - piv_origins[:,1]  # Subtract from uppermost pixel to get ground coordinate
    piv_vectors = np.asarray(piv_vectors)
    piv_vectors *= geo_transform[0,0]  # Scale by pixel ground size
    peak_covariance = np.asarray(peak_covariance)
    peak_covariance *= geo_transform[0,0]**2  # Scale by squared pixel ground size

    piv_end_location = piv_origins
    piv_end_location[:,0] += piv_vectors[:,0]     
    piv_end_location[:,1] -= piv_vectors[:,1]  # Subtract to convert from dV (positive down) to dY (positive up)

    piv_end_location = piv_end_location.tolist()
    peak_covariance = peak_covariance.tolist()
    locations_covariances = []
    for i in range(len(piv_end_location)):
        locations_covariances.append([piv_end_location[i], peak_covariance[i]])

    json.dump(locations_covariances, open(output_base_name + "covariances.json", "w"))
    print("PIV covariance matrices saved to file '{}covariances.json'".format(output_base_name))


def get_bias_variance(output_base_name):
    with open(output_base_name + "vectors.json", "r") as json_file:
        origins_vectors = json.load(json_file)
    origins_vectors = np.asarray(origins_vectors)

    x_bias_variance = np.var(origins_vectors[:,2])
    y_bias_variance = np.var(origins_vectors[:,3])

    return [x_bias_variance, y_bias_variance]


def add_bias_variance(output_base_name, xy_bias_variance):
    with open(output_base_name + "covariances.json", "r") as json_file:
        covariance_matrices = json.load(json_file)
    
    for i in range(len(covariance_matrices)):
        covariance_matrices[i][1][0][0] += xy_bias_variance[0]
        covariance_matrices[i][1][1][1] += xy_bias_variance[1]
    
    json.dump(covariance_matrices, open(output_base_name + "covariances.json", "w"))