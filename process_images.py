import os
import helpers
from skimage import io, filters, morphology, measure, draw, feature, restoration
import pandas as pd
from pybaselines import Baseline2D
import numpy as np
from matplotlib import pyplot as plt

def fill_missing_with_median(image):
    filled_image = image.copy()
    nan_mask = np.isnan(filled_image)  # Mask for missing pixels
    
    while np.any(nan_mask):
        median_filled = filters.median(np.nan_to_num(filled_image, nan=np.nanmedian(filled_image)), footprint=morphology.disk(3))
        filled_image[nan_mask] = median_filled[nan_mask]
        nan_mask = np.isnan(filled_image)

    return filled_image

def filter_bubbles(image):
    while np.count_nonzero(image==0) >1000:
        edge_scharr = filters.scharr(image)
        edge_scharr_bw = edge_scharr > 0.05 * edge_scharr.max()
        image_copy = np.copy(image.astype(float))
        image_copy[edge_scharr_bw] = np.nan
        image_filled = fill_missing_with_median(image_copy)
        image = np.copy(image_filled.astype(np.uint16))
    return image

def correct_image(file, lam=(1e6, 1e6)):
    signal = io.imread(file, plugin = 'pil')
    membrane, memCoverage = helpers.find_membrane(signal)
    cropped_sig = signal[np.where(membrane)[0].min():np.where(membrane)[0].max(), 
                             np.where(membrane)[1].min():np.where(membrane)[1].max()]
    
    if cropped_sig.shape[0] % 2 == 1: cropped_sig = np.delete(cropped_sig, cropped_sig.shape[0] - 1, axis=0)
    if cropped_sig.shape[1] % 2 == 1: cropped_sig = np.delete(cropped_sig, cropped_sig.shape[1] - 1, axis=1)
    
    corr_signal = helpers.denoiseImage(cropped_sig)
    corr_signal = filters.median(corr_signal, footprint = morphology.disk(7))
    
    # edge_scharr = filters.scharr(corr_signal)
    # point_y, point_x = np.where((edge_scharr > 0.05*edge_scharr.max()))

    corr_signal_filled = filter_bubbles(corr_signal)
    x = np.arange(corr_signal_filled.shape[0])
    z = np.arange(corr_signal_filled.shape[1])
    
    baseline_fitter = Baseline2D(x, z, check_finite=False) 
    # baseline, params = baseline_fitter.arpls(corr_signal_filled, lam=(1e6, 1e6 ) )
    baseline, _ = baseline_fitter.arpls(corr_signal_filled, lam=lam )
    
    corrected_sig = corr_signal - baseline
    corrected_sig[corrected_sig > corr_signal.max()] = 0
    return cropped_sig, baseline, corrected_sig

def generate_peak_mask_old(che_img, peak_indices, lvl = None, row_offset = 75, col_offset = 75):
    if lvl == None: lvl = 20*np.median(che_img)
    che_mask = np.zeros(che_img.shape, dtype=np.bool_)
    for i in range(len(peak_indices)):
        peak_img = che_img[peak_indices[i][0] - row_offset : peak_indices[i][0] + row_offset ,
                           peak_indices[i][1] - col_offset : peak_indices[i][1] + col_offset]
        peak_mask = np.zeros(peak_img.shape, dtype = np.bool_)
        contours = measure.find_contours(peak_img, lvl)
        for contour in contours:
            if (contour[0] == contour[-1]).all():
                rows, cols = draw.polygon(contour[:, 0], contour[:, 1])
                peak_mask[rows, cols] = True
        che_mask[peak_indices[i][0]-row_offset:peak_indices[i][0]+row_offset , 
                 peak_indices[i][1]-col_offset:peak_indices[i][1]+col_offset] = peak_mask
    return che_mask

def generate_peak_mask(fun_img, min_dist = 100):
    # peaks = feature.corner_peaks(fun_img, indices=True, min_distance=min_dist, num_peaks=80)
    peaks = feature.peak_local_max(fun_img, min_distance=min_dist, num_peaks=80)
    level = np.median(fun_img)
    contours = measure.find_contours(fun_img, level)
    found_contours = []
    # max_iteration = 0
    while len(found_contours) < len(peaks): # - 5 and max_iteration < 30:
        prev_contours = found_contours
        found_contours = []
        contours = measure.find_contours(fun_img, level)
        for contour in contours:
            if measure.points_in_poly(peaks, contour).any() == True:
                found_contours.append(contour)
        if len(found_contours) < len(prev_contours):
            found_contours = prev_contours
            break
        level += 1
        # max_iteration += 1

        # plt.imshow(fun_img, cmap='gray')
        # for contour in found_contours:
        #     plt.plot(contour[:, 1], contour[:, 0], linewidth=1)
        # plt.show()
    fun_mask = plot_mask(found_contours, fun_img)
    # fun_mask = np.zeros(fun_img.shape, dtype=np.bool_)
    # for contour in found_contours:
    #     rows, cols = draw.polygon(contour[:, 0], contour[:, 1])
    #     fun_mask[rows, cols] = True

    # helpers.showRoiBoundary(fun_img, fun_mask, file_= fname)
    # plt.show()
    return fun_mask

def band_finder(corrected_signal, min_dist = 100, level = None):
    peaks = feature.peak_local_max(corrected_signal, min_distance=min_dist, num_peaks=80)
    if level == None: level = np.median(corrected_signal)
    found_contours = []
    while len(found_contours) < len(peaks):
        prev_contours = found_contours.copy() #.copy is new in this function from generate_peak_mask function
        found_contours = []
        contours = measure.find_contours(corrected_signal, level)
        for contour in contours:
            if measure.points_in_poly(peaks, contour).any() == True:
                found_contours.append(contour)
        if len(found_contours) < len(prev_contours):
            found_contours = prev_contours.copy()
            missing_peaks = peaks.copy()
            for contour in found_contours:
                found_peaks = missing_peaks[np.where(measure.points_in_poly(missing_peaks, contour) == True)]
                for peak in found_peaks:
                    matching_index = np.where((missing_peaks == peak).all(axis=1))[0]
                    missing_peaks = np.delete(missing_peaks, matching_index, axis=0)
            break
        level += 1
    
    return level, missing_peaks, found_contours


def plot_mask(found_contours, corrected_sig):
    fun_mask = np.zeros(corrected_sig.shape, dtype=np.bool_)
    for contour in found_contours:
        if (contour[0] != contour[-1]).all():
            R_max, C_max = corrected_sig.shape
            Rs, Re, Cs, Ce = contour[0][0], contour[-1][0], contour[0][1], contour[-1][1]
            # start_pt = contour[0]
            # end_pt = contour[-1]
            points = np.array([Rs, Re, Cs, Ce])
            CP = []
            if ((points[:2] == 0).any() and (points[:2] == R_max-1).any()) or ((points[2:] == 0).any() and (points[2:] == C_max-1).any()):
                closed_contour = None
                break
            else:
                if (points[:2] == 0).any():
                    CP.append(0)
                if (points[:2] == R_max-1).any():
                    CP.append(R_max-1)
                if (points[2:] == 0).any():
                    CP.append(0)
                if (points[2:] == C_max-1).any():
                    CP.append(C_max-1)
            if len(CP) == 2:
                closed_contour = np.vstack([contour, CP])
            else:
                closed_contour = contour
        else:
            closed_contour = contour

        rows, cols = draw.polygon(closed_contour[:, 0], closed_contour[:, 1])
        fun_mask[rows, cols] = True
    return fun_mask

def calibrated_denoiser (cropped_sig):
    import numpy as np
    parameters = {'weight': np.arange(0.01, 0.3, 0.02),
                  'wavelet': ['db1', 'db4', 'bl9', 'coif4', 'sym6', 'vaid'],
                  'sigma': np.arange(0.1, 0.3, 0.02)
                  }
    denoising_function = restoration.calibrate_denoiser(cropped_sig, restoration.denoise_wavelet,
                                            denoise_parameters=parameters)
    corrected_sig = denoising_function(cropped_sig)
    return corrected_sig