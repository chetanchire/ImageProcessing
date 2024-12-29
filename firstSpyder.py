# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:18:09 2024

@author: Chetan Hire
"""
import helpers
from helpers import find_membrane, io, showRoiBoundary

signal = io.imread("2024_10_23_143655 30.tif")
signal = helpers.denoiseImage(signal)
membrane, memCoverage = find_membrane(signal)
showRoiBoundary(signal, membrane)
corr_signal = helpers.baseline_corrected_img(signal, membrane)
helpers.showImgProfile(corr_signal)
helpers.showImgProfile(signal)
corr_signal_arpls = helpers.baseline_corrected_img(signal, membrane, method=1)
helpers.showImgProfile(corr_signal_arpls, eqHist=2)
helpers.show3DImg(corr_signal_arpls)

from pybaselines import Baseline2D
import numpy as np

cropped_sig = signal[np.where(membrane)[0].min():np.where(membrane)[0].max(), 
                     np.where(membrane)[1].min():np.where(membrane)[1].max()]

from skimage import filters
smooth_sig = filters.gaussian(cropped_sig, sigma=100, preserve_range = True)

x=np.arange(smooth_sig.shape[0])
z=np.arange(smooth_sig.shape[1])

baseline_fitter = Baseline2D(x, z, check_finite=False)
baseline, params = baseline_fitter.arpls(smooth_sig, lam=(1e6, 1e6))
corrected_image = cropped_sig.astype(np.float32)-baseline
corrected_image = corrected_image - corrected_image.min()
helpers.show3DImg(corrected_image, zlim=30000)
helpers.showImgProfile(corrected_image)
helpers.showImgProfile(cropped_sig)
helpers.showImgProfile(baseline)
helpers.show3DImg(baseline, zlim=30000)
helpers.show3DImg(cropped_sig, zlim=30000)

signal = io.imread("C:/Users/Chetan Hire/OneDrive - proteowise.com/Share with Gary/2024-06-12 101428/improc/signal/015 - SRF #5147 Signal.tif")
from skimage import filters, morphology, exposure
import numpy as np
fil_sig = filters.median(signal, footprint = morphology.disk(7))
rescale_sig = exposure.rescale_intensity(fil_sig, out_range=(0, 60000))
io.imsave('edited_img.tif', rescale_sig.astype(np.uint16))
