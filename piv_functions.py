import sys
import math
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from skimage.feature import match_template, peak_local_max
from scipy import interpolate, ndimage
from scipy.spatial.distance import cdist
import rasterio

import show_functions
from robust_smooth_2d import robust_smooth_2d


def format_input(before,
                 after,
                 template_size,
                 step_size,
                 prop,
                 outname):

    user_input = dict(before_file=before,
                      after_file=after,
                      before_uncertainty_file="",
                      after_uncertainty_file="",
                      template_size=template_size,
                      step_size=step_size,
                      propagate=False,
                      output_base_name="")

    if prop:
        user_input["propagate"] = True
        user_input["before_uncertainty_file"] = prop[0]
        user_input["after_uncertainty_file"] = prop[1]

    if outname:
        user_input["output_base_name"] = outname + "_"

    # Silently force odd size template for clarity in PIV computations
    if template_size % 2 == 0:
        user_input["template_size"] = template_size + 1

    return user_input


def ingest_data(user_input):
    image_data = dict(before=[],
                      after=[],
                      before_uncertainty=[],
                      after_uncertainty=[],
                      geo_transform=[])

    source = rasterio.open(user_input["before_file"])
    image_data["before"] = source.read(1)
    before_geo_transform = np.reshape(np.asarray(source.transform), (3,3))
    source = rasterio.open(user_input["after_file"])
    image_data["after"] = source.read(1)
    after_geo_transform = np.reshape(np.asarray(source.transform), (3,3))

    # PIV results are misleading if the before and after data is not on the 
    # same datum and of equal spatial extent
    if not np.array_equal(before_geo_transform, after_geo_transform):
        print("The spatial extent and/or datum of the 'before' and 'after' ",
              "data is not equivalent.")
        sys.exit()
    else:
        image_data["geo_transform"] = before_geo_transform

    if user_input["propagate"]:
        image_data["before_uncertainty"] = rasterio.open(
            user_input["before_uncertainty_file"]).read(1)
        image_data["after_uncertainty"] = rasterio.open(
            user_input["after_uncertainty_file"]).read(1)

    return image_data


def run_piv(user_input, image_data):
    """
    An iterative PIV approach is used to handle the presence of shear, or 
    gradient, in the movement between the before and after data. During each 
    iteration, the computed PIV vectors are used to deform the 'after' data.
    We use an object to store the current and cumulative PIV vectors during the
    iterative process. Three iterations should be sufficient to remove shear.
    The 'after' data is not deformed on the final iteration, since the current
    deformation algorithm smooths the PIV vectors (we do not want the final 
    result smoothed).

    If uncertainty is being propagated, we need to estimate the bias variance
    and add it to the propagated variances. The bias variance is the spatial
    variance in the PIV vector components that is present when PIV is run on
    two identical images.
    """
    piv = Piv(image_data)

    # Start with two PIV iterations with no uncertainty propagation
    # for i in range(2):
    #     piv.correlate(user_input["template_size"],
    #                   user_input["step_size"],
    #                   False)
    #     piv.deform(user_input["template_size"],
    #                user_input["step_size"],
    #                user_input["propagate"])
    # For the third (final) iteration, propagate uncertainty if requested and
    # compute the final, cumulative vector displacements
    piv.correlate(user_input["template_size"],
                  user_input["step_size"],
                  user_input["propagate"])
    piv.compute_total()

    # Estimate and add bias variance if propagating uncertainty
    if user_input["propagate"]:
        estimate_bias(piv, user_input, image_data)

    # All done. Export the PIV vectors and uncertainties to JSON files
    piv.export(user_input, image_data)
    if user_input["propagate"]:
        piv.export_uncertainty(user_input)

    # Show results    
    if user_input["propagate"]:
        show_functions.show(
            user_input["before_file"],
            vector_file=user_input["output_base_name"] + 'vectors.json',
            covariance_file=user_input["output_base_name"] + 'covariances.json',
        )
    else:
        show_functions.show(
            user_input["before_file"],
            vector_file=user_input["output_base_name"] + 'vectors.json'
        )


def estimate_bias(piv, user_input, image_data):
    temp = image_data["after"]
    image_data["after"] = image_data["before"]
    piv_bias = Piv(image_data)
    piv_bias.correlate(user_input["template_size"],
                       user_input["step_size"],
                       False)
    image_data["after"] = temp
    u_bias_variance, v_bias_variance = piv_bias.compute_bias()
    piv.add_bias(u_bias_variance, v_bias_variance)


class Piv:
    def __init__(self, image_data):
        self._before = image_data["before"]
        self._before_uncertainty = image_data["before_uncertainty"]
        self._after = image_data["after"]
        self._after_uncertainty = image_data["after_uncertainty"]
        self._after_deformed = image_data["after"]        
        self._after_uncertainty_deformed = image_data["after_uncertainty"]
        self._deformation_field_u = np.zeros(image_data["after"].shape)
        self._deformation_field_u_total = np.zeros(image_data["after"].shape)
        self._deformation_field_v = np.zeros(image_data["after"].shape)
        self._deformation_field_v_total = np.zeros(image_data["after"].shape)
        self._delta = 0.000001  # Partial derivative increment
        self._piv_vectors = []
        self._piv_origins = []
        self._final_vectors = []
        self._peak_covariance = []


    def correlate(self, template_size, step_size, propagate):
        # Housecleaning
        self._piv_origins = []
        self._piv_vectors = []

        # Progress display prep
        status_figure = plt.figure()
        before_axis = plt.subplot(1, 2, 1)
        after_axis = plt.subplot(1, 2, 2)

        # Pre-extract all the required window data
        window_data = self._get_windows(template_size, step_size, propagate)

        for record in window_data:
            # Display progress
            self._show_piv_location(before_axis, after_axis,
                                    record[1][0], record[1][1],
                                    record[2][0], record[2][1],
                                    template_size, template_size*2)

            # Compute normalized cross-correlation (NCC)
            ncc = match_template(record[4], record[3])
            max_idx = np.squeeze(np.asarray(np.where(ncc == np.max(ncc))))

            # Check for peak location on edges of NCC matrix - this breaks
            # the sub-pixel peak interpolation
            if (max_idx[0]==0 or 
                    max_idx[1]==0 or
                    max_idx[0]==ncc.shape[0]-1 or
                    max_idx[1]==ncc.shape[1]-1):
                continue

            # Solve sub-pixel NCC peak
            subpixel_peak = self._subpixel_peak(ncc, max_idx)

            # Store PIV origin and vector
            self._store_origin_vector(record[0], max_idx,
                                      template_size, subpixel_peak)

            # Propagate uncertainty if requested and store
            if propagate:
                self._propagate_uncertainty(record, ncc, max_idx,
                                            subpixel_peak, template_size)

        # All done. Close progress display figure
        plt.close(status_figure)


    def deform(self, template_size, step_size, propagate):
        # Filter outlier PIV vectors
        filtered_origins, filtered_vectors = self._filter_vectors(
            template_size,
            step_size
        )

        # Interpolate grid of vectors for each DEM pixel
        self._interpolate_vectors(
            filtered_origins,
            filtered_vectors
        )

        # Deform the 'After' DEM and uncertainty map
        self._deform_images(propagate)


    def compute_total(self):
        for i in range(len(self._piv_origins)):
            # cumulative piv vectors from iterations prior to final iteration
            u_cumulative = self._deformation_field_u_total[
                self._piv_origins[i][1],
                self._piv_origins[i][0]
            ]
            v_cumulative = self._deformation_field_v_total[
                self._piv_origins[i][1],
                self._piv_origins[i][0]
            ]
            # Update PIV vectors with cumulative vectors from prior iterations
            self._piv_vectors[i][0] += u_cumulative
            self._piv_vectors[i][1] += v_cumulative

        # # Status
        # piv_origins = np.asarray(self._piv_origins)
        # piv_vectors = np.asarray(self._piv_vectors)
        # plt.quiver(piv_origins[:,0], -piv_origins[:,1], piv_vectors[:,0], piv_vectors[:,1],angles='xy',scale_units='xy')
        # plt.axis('equal')
        # plt.show()


    def compute_bias(self):
        piv_vectors = np.asarray(self._piv_vectors)
        u_bias_variance = np.var(piv_vectors[:,0])
        v_bias_variance = np.var(piv_vectors[:,1])
        return [u_bias_variance, v_bias_variance]


    def add_bias(self, u_bias_variance, v_bias_variance):
        for covariance in self._peak_covariance:
            covariance[0][0] += u_bias_variance
            covariance[1][1] += v_bias_variance


    def export(self, user_input, image_data):
        # Convert from pixels to ground distance
        piv_origins = np.asarray(self._piv_origins, dtype=np.float64)
        piv_origins *= image_data["geo_transform"][0,0]  # Scale by pixel ground size
        piv_origins[:,0] += image_data["geo_transform"][0,2]  # Offset by leftmost pixel to get ground coordinate
        piv_origins[:,1] = image_data["geo_transform"][1,2] - piv_origins[:,1]  # Subtract from uppermost pixel to get ground coordinate    
        piv_vectors = np.asarray(self._piv_vectors)
        piv_vectors *= image_data["geo_transform"][0,0]  # Scale by pixel ground size
        
        origins_vectors = np.concatenate((piv_origins, piv_vectors), axis=1)
        json.dump(origins_vectors.tolist(), open(user_input["output_base_name"] + "vectors.json", "w"))
        print("PIV displacement vectors saved to file '{}vectors.json'".format(user_input["output_base_name"]))


    def export_uncertainty(input_data, image_data):
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


    def _get_windows(self, template_size, step_size, propagate):
        search_size = template_size*2
        num_hz_comps = math.floor((self._before.shape[1] - search_size)
                                   / step_size)
        num_vt_comps = math.floor((self._before.shape[0] - search_size)
                                   / step_size)

        window_data = []
        for vt in range(num_vt_comps):
            for hz in range(num_hz_comps):
                # Correlation origin
                origin = [hz*step_size + template_size, 
                          vt*step_size + template_size]

                # Template hz and vt start indices
                hz_template_start = int(hz*step_size + (template_size+1)/2)
                vt_template_start = int(vt*step_size + (template_size+1)/2)
                # Template hz and vt end indices
                hz_template_end = int(hz_template_start + template_size - 1)
                vt_template_end = int(vt_template_start + template_size - 1)
                # Search hz and vt start indices
                hz_search_start = int(hz*step_size)
                vt_search_start = int(vt*step_size)
                # Search hz and vt end indices
                hz_search_end = int(hz_search_start + search_size)
                vt_search_end = int(vt_search_start + search_size)

                # Template and search image patches
                template = self._before[vt_template_start:vt_template_end,
                    hz_template_start:hz_template_end].copy()
                search = self._after_deformed[vt_search_start:vt_search_end,
                    hz_search_start:hz_search_end].copy()
                # Template and search uncertainty image patches
                if propagate:
                    template_uncertainty = self._before_uncertainty[
                        vt_template_start:vt_template_end,
                        hz_template_start:hz_template_end].copy()
                    search_uncertainty = self._after_uncertainty_deformed[
                        vt_search_start:vt_search_end,
                        hz_search_start:hz_search_end].copy()
                else:
                    template_uncertainty = []
                    search_uncertainty = []

                # Guard against flat patches; they cause a divide by zero error
                if (np.max(template)-np.min(template) < 1e-10 or
                        np.max(search)-np.min(search) < 1e-10):
                    continue
                # Guard against NaN values; they break SciPy's match_template()
                if (np.isnan(template).any() or np.isnan(search).any()):
                    continue

                # Append
                window_data.append([
                    origin,
                    [hz_template_start, vt_template_start],
                    [hz_search_start, vt_search_start],
                    template,
                    search,
                    template_uncertainty,
                    search_uncertainty])

        return window_data


    def _show_piv_location(self,
                           before_axis, after_axis,
                           hz_template_start, vt_template_start,
                           hz_search_start, vt_search_start,
                           template_size, search_size):
        plt.sca(before_axis)
        plt.cla()
        before_axis.set_title('Before')
        before_axis.imshow(self._before, cmap=plt.cm.gray)
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
        after_axis.imshow(self._after_deformed, cmap=plt.cm.gray)
        after_axis.add_patch(matplotlib.patches.Rectangle(
            (hz_search_start,vt_search_start), 
            search_size-1, 
            search_size-1, 
            linewidth=1, 
            edgecolor='r',
            fill=None))

        plt.pause(0.1)


    def _subpixel_peak(self, ncc, max_idx):
        # Extract 3x3 submatrix centered on NCC maximum pixel
        local = ncc[max_idx[0]-1:max_idx[0]+2, 
                    max_idx[1]-1:max_idx[1]+2]

        du = (local[1,2] - local[1,0]) / 2
        duu = local[1,2] + local[1,0] - 2*local[1,1]
        dv = (local[2,1] - local[0,1]) / 2
        dvv = local[2,1] + local[0,1] - 2*local[1,1]
        duv = (local[2,2] - local[2,0] - local[0,2] + local[0,0]) / 4
        
        # u_delta is positive left-to-right; v_delta is positive top-to-bottom
        u_delta = -(dvv*du - duv*dv) / (duu*dvv - duv*duv)
        v_delta = -(duu*dv - duv*du) / (duu*dvv - duv*duv)

        return [u_delta, v_delta]


    def _store_origin_vector(self, origin, max_idx,
                             template_size, subpixel_peak):
        self._piv_origins.append(origin)
        self._piv_vectors.append(
            [max_idx[1] - (template_size+1)/2 + subpixel_peak[0],
             max_idx[0] - (template_size+1)/2 + subpixel_peak[1]])


    def _propagate_uncertainty(self, record, ncc, max_idx,
                               subpixel_peak, template_size):
        # Propagate raster error into the 3x3 patch of correlation values that 
        # are centered on the correlation peak
        correlation_covariance = self._propagate_pixel_into_correlation(
            record[3],
            record[5], 
            # templateSize+2 x templateSize+2 subarray of the search array
            record[4][max_idx[0]-1:max_idx[0]+template_size+1,
                      max_idx[1]-1:max_idx[1]+template_size+1],
            # templateSize+2 x templateSize+2 subarray of the search uncertainty
            record[6][max_idx[0]-1:max_idx[0]+template_size+1,
                      max_idx[1]-1:max_idx[1]+template_size+1],
            # 3x3 array of correlation values centered on the correlation peak
            ncc[max_idx[0]-1:max_idx[0]+2,
                max_idx[1]-1:max_idx[1]+2], 
            self._delta) 

        # Propagate the correlation covariance into the subpixel peak location
        peak_covariance = self._propagate_correlation_into_subpixel_peak(
            ncc[max_idx[0]-1:max_idx[0]+2, max_idx[1]-1:max_idx[1]+2],
            correlation_covariance,
            subpixel_peak,
        )

        # All done. Store.
        self._peak_covariance.append(peak_covariance.tolist())


    def _propagate_pixel_into_correlation(self,
                                          height_template,
                                          uncertainty_template, 
                                          height_search,
                                          uncertainty_search,
                                          normalized_cross_correlation,
                                          numeric_delta):
        template_covariance_vector = np.square(uncertainty_template.reshape(uncertainty_template.size,)) # convert array to vector, row-by-row, and square the standard deviations into variances
        search_covariance_vector = np.square(uncertainty_search.reshape(uncertainty_search.size,))
        covariance_matrix = np.diag(np.hstack((template_covariance_vector, search_covariance_vector)))

        jacobian = self._get_correlation_jacobian(height_template, height_search, normalized_cross_correlation, numeric_delta)    
        # Propagate the template and search area errors into the 9 correlation elements
        # The covariance order is by row of the normalized_cross_correlation (ncc) array (i.e., ncc[0,0], ncc[0,1], ncc[0,2], ncc[1,0], ncc[1,1], ...)
        correlation_covariance = np.matmul(jacobian,np.matmul(covariance_matrix,jacobian.T))

        return correlation_covariance

    def _get_correlation_jacobian(self, template, search,
                                  normalized_cross_correlation,
                                  numeric_delta):

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
                        perturbed_template[row_template,col_template] += numeric_delta
                        perturbed_search_subarea = search_subarea.copy()
                        perturbed_search_subarea[row_template,col_template] += numeric_delta

                        # spatial domain normalized cross correlation (i.e., does not use the FFT)
                        # about 20x faster than using skimage's FFT based match_template method
                        normalized_perturbed_template = (perturbed_template - np.mean(perturbed_template)) / (np.std(perturbed_template))
                        normalized_perturbed_search_subarea = (perturbed_search_subarea - np.mean(perturbed_search_subarea)) / (np.std(perturbed_search_subarea))
                        perturbed_template_normalized_cross_correlation = np.sum(normalized_perturbed_template * normalized_search_subarea) / template.size
                        perturbed_search_subarea_normalized_cross_correlation = np.sum(normalized_template * normalized_perturbed_search_subarea) / template.size
                        
                        # storage location adjustment by row_correlation and col_correlation accounts for the larger size of the search area than the template area
                        template_partial_derivatives[row_template, col_template] = (perturbed_template_normalized_cross_correlation - normalized_cross_correlation[row_correlation,col_correlation]) / numeric_delta
                        search_partial_derivatives[row_correlation+row_template, col_correlation+col_template] = (perturbed_search_subarea_normalized_cross_correlation - normalized_cross_correlation[row_correlation, col_correlation]) / numeric_delta 

                # reshape the partial derivatives from their current array form to vector form and store in the Jacobian
                # we match the row-by-row pattern used to form the covariance matrix in the calling function
                jacobian[row_correlation*3+col_correlation, 0:template.size] = template_partial_derivatives.reshape(template_partial_derivatives.size,)
                jacobian[row_correlation*3+col_correlation, template.size:template.size+search.size] = search_partial_derivatives.reshape(search_partial_derivatives.size,)

        return jacobian

    def _propagate_correlation_into_subpixel_peak(self,
                                                  correlation,
                                                  correlation_covariance,
                                                  subpixel_peak):
        jacobian = np.zeros((2,9))
        # Cycle through the 3x3 correlation array, row-by-row, and create the
        # jacobian matrix
        for row in range(3):
            for col in range(3):
                perturbed_correlation = correlation.copy()
                perturbed_correlation[row,col] += self._delta
                hz_delta, vt_delta = self._subpixel_peak(perturbed_correlation,
                                                         [1, 1])
                jacobian[0,row*3+col] = ((hz_delta - subpixel_peak[0])
                                          / self._delta)
                jacobian[1,row*3+col] = ((vt_delta - subpixel_peak[1])
                                          / self._delta)
        # Propagate the 3x3 array of correlation uncertainties into the 
        # sub-pixel u and v PIV vector components
        subpixel_peak_covariance = np.matmul(
            jacobian,
            np.matmul(correlation_covariance, jacobian.T)
        )

        return subpixel_peak_covariance


    def _filter_vectors(self, template_size, step_size):
        # Create a NaN image for each vector component (u and v)
        search_size = template_size*2
        num_hz_comps = math.floor((self._before.shape[1] - search_size)
                                   / step_size)
        num_vt_comps = math.floor((self._before.shape[0] - search_size)
                                   / step_size)
        u_img = np.empty((num_vt_comps, num_hz_comps))
        u_img[:] = np.nan
        v_img = u_img.copy()
        # Overwrite the NaN values where valid PIV vectors exist
        for i in range(len(self._piv_origins)):
            row = int((self._piv_origins[i][1] - template_size)/step_size)
            col = int((self._piv_origins[i][0] - template_size)/step_size)
            u_img[row,col] = self._piv_vectors[i][0]
            v_img[row,col] = -self._piv_vectors[i][1]
        # Smooth the u and v vector component images
        u_smooth, s = robust_smooth_2d(u_img, robust=True) #s=0.00005
        v_smooth, s = robust_smooth_2d(v_img, robust=True)

        # Transform origins and vectors back to a list format
        filtered_origins = []
        filtered_vectors = []
        for vt_count in range(num_vt_comps):
            for hz_count in range(num_hz_comps):
                filtered_origins.append([hz_count*step_size + template_size,
                                         vt_count*step_size + template_size])
                filtered_vectors.append([u_smooth[vt_count, hz_count],
                                         v_smooth[vt_count, hz_count]])

        # Status figure of before and after smoothing
        status_figure = plt.figure()
        u_axis = plt.subplot(2, 3, 1)
        v_axis = plt.subplot(2, 3, 2)
        u_axis.imshow(u_img)
        v_axis.imshow(v_img)
        before_axis = plt.subplot(2, 3, 3)
        piv_origins = np.asarray(self._piv_origins)
        piv_vectors = np.asarray(self._piv_vectors)
        before_axis.quiver(piv_origins[:,0], -piv_origins[:,1], piv_vectors[:,0], -piv_vectors[:,1], angles='xy', scale_units='xy')
        before_axis.axis('equal') 
        u_axis_smooth = plt.subplot(2, 3, 4)
        v_axis_smooth = plt.subplot(2, 3, 5)
        u_axis_smooth.imshow(u_smooth)
        v_axis_smooth.imshow(v_smooth)
        after_axis = plt.subplot(2, 3, 6)
        temp_piv_origins = []
        temp_piv_u = []
        temp_piv_v = []
        for vt_count in range(num_vt_comps):
            for hz_count in range(num_hz_comps):
                temp_piv_origins.append([hz_count*step_size + template_size,
                                         vt_count*step_size + template_size])
                row = vt_count
                col = hz_count
                temp_piv_u.append(u_smooth[row, col])
                temp_piv_v.append(v_smooth[row, col])
        temp_piv_origins = np.asarray(temp_piv_origins)
        temp_piv_u = np.asarray(temp_piv_u)
        temp_piv_v = np.asarray(temp_piv_v)
        after_axis.quiver(temp_piv_origins[:,0], -temp_piv_origins[:,1], temp_piv_u, temp_piv_v, angles='xy', scale_units='xy')
        plt.show()

        return filtered_origins, filtered_vectors


    def _interpolate_vectors(self, filtered_origins, filtered_vectors):
        # Cubic interpolation of vector components for each pixel. SciPy's
        # interpolate.interp2d is deliberately not used (unreliable results).
        piv_origins = np.asarray(filtered_origins)
        piv_vectors = np.asarray(filtered_vectors)
        u_vec = np.arange(self._after.shape[1])
        v_vec = np.arange(self._after.shape[0])
        u_grid, v_grid = np.meshgrid(u_vec, v_vec)
        cubic_u = interpolate.griddata((piv_origins[:,0], piv_origins[:,1]), piv_vectors[:,0], (u_grid, v_grid), method="cubic")
        cubic_v = interpolate.griddata((piv_origins[:,0], piv_origins[:,1]), piv_vectors[:,1], (u_grid, v_grid), method="cubic")        
        # Fill missing interpolation data (on outsides) with nearest values
        is_finite = np.isfinite(cubic_u)
        nearest_u = interpolate.griddata((u_grid[is_finite], v_grid[is_finite]), cubic_u[is_finite], (u_grid, v_grid), method="nearest")
        nearest_v = interpolate.griddata((u_grid[is_finite], v_grid[is_finite]), cubic_v[is_finite], (u_grid, v_grid), method="nearest")
        cubic_u[~is_finite] = nearest_u[~is_finite]
        cubic_v[~is_finite] = nearest_v[~is_finite]
        # Save results
        self._deformation_field_u = cubic_u
        self._deformation_field_v = cubic_v
        self._deformation_field_u_total += self._deformation_field_u
        self._deformation_field_v_total += self._deformation_field_v

        # status figure
        status_figure = plt.figure()
        axu = plt.subplot(1,3,1)
        axu.imshow(self._deformation_field_u)
        axv = plt.subplot(1,3,2)
        axv.imshow(self._deformation_field_v)
        axvt = plt.subplot(1,3,3)
        axvt.imshow(self._deformation_field_v_total)
        plt.show()
        # status figure
        # status_figure = plt.figure()
        # computed_axis = plt.subplot(1, 2, 1)
        # interpolated_axis = plt.subplot(1, 2, 2)
        # computed_axis.quiver(piv_origins[:,0], -piv_origins[:,1], piv_vectors[:,0], piv_vectors[:,1],angles='xy',scale_units='xy')
        # computed_axis.axis('equal')
        # u_vec, v_vec = np.meshgrid(np.arange(self._after.shape[1]), np.arange(self._after.shape[0]))
        # interpolated_axis.quiver(u_vec[::20,::20],-v_vec[::20,::20],self._deformation_field_u[::20,::20],self._deformation_field_v[::20,::20],angles='xy',scale_units='xy')
        # interpolated_axis.axis('equal')
        # plt.show()


    def _deform_images(self, propagate):
         # Deform the 'After' DEM
        image_u_coords, image_v_coords = np.meshgrid(
            np.arange(self._after.shape[1]),
            np.arange(self._after.shape[0])
        )
        # u_coord = image_u_coords + self._deformation_field_u_total
        # v_coord = image_v_coords - self._deformation_field_v_total
        u_coord = image_u_coords + self._deformation_field_u
        v_coord = image_v_coords - self._deformation_field_v
        # self._after_deformed = ndimage.map_coordinates(
        #     self._after,
        #     [v_coord.ravel(), u_coord.ravel()],
        #     order=3,
        #     mode='nearest'
        # ).reshape(self._after.shape)
        self._after_deformed = ndimage.map_coordinates(
            self._after_deformed,
            [v_coord.ravel(), u_coord.ravel()],
            order=3,
            mode='nearest'
        ).reshape(self._after.shape)
        
        # Deform the 'After' uncertainty image if propagating uncertainty
        if propagate:
            self._after_uncertainty_deformed = ndimage.map_coordinates(
                self._after_uncertainty,
                [v_coord.ravel(), u_coord.ravel()],
                order=3,
                mode='nearest'
            ).reshape(self._after_uncertainty.shape)

        # status_figure = plt.figure()
        # original_axis = plt.subplot(1, 3, 1)
        # deformed_axis = plt.subplot(1, 3, 3)
        # before_axis = plt.subplot(1, 3, 2)
        # original_axis.set_title('Original After')
        # original_axis.imshow(self._after, cmap=plt.cm.gray)
        # deformed_axis.set_title('Deformed After')
        # deformed_axis.imshow(self._after_deformed, cmap=plt.cm.gray)
        # before_axis.set_title('Before')
        # before_axis.imshow(self._before, cmap=plt.cm.gray)
        # plt.show()