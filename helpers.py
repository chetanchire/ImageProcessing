# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:23:39 2024

@author: Chetan Hire
"""

from matplotlib import pyplot as plt
import numpy as np
from skimage import io, morphology, exposure, measure, draw, color, filters, segmentation, restoration, metrics, img_as_float
import pywt
from scipy import signal, sparse, ndimage
import warnings
warnings.filterwarnings('ignore')
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
import os
from pybaselines import Baseline2D

def getVarName(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None

def showImgHist(Image, log=1, eqHist=1, cMap = 'gray'):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24,8), width_ratios=(1,1.5))
    Image_eq = exposure.equalize_hist(Image) if eqHist==1 else Image
    axs[0].imshow(Image_eq, cmap=cMap); 
    axs[0].set_title(getVarName(Image))
    axs[1].hist(Image.ravel(), bins = 256, log = log)
    return fig.tight_layout

def showRoiBoundary(Image, Mask, eqHist=1):
    if eqHist == 1:
        Image_eq = exposure.equalize_hist(Image) 
    elif eqHist == 2:
        Image_eq = exposure.equalize_adapthist(Image)
    else:
        Image_eq = Image
    Image_eq = exposure.equalize_hist(Image) if eqHist==1 else Image
    if Mask.dtype == np.bool_ and Image.dtype == np.uint16:
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(6, 6))
        ax.imshow(Image_eq, cmap='gray')
        # for contour in contours:
            # ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax.contour(Mask, colors='red', linewidths=1)
        ax.set_title(f'{getVarName(Mask)} over {getVarName(Image)}')
        return fig.tight_layout
    else:
        raise SystemExit('Images dont match required data type. First argument is uint8 image and second argument is a binary image')

def show3DImg (signal, zlim = None):
    #signal = io.imread("signal.tif", plugin='pil')
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')
    x = np.arange(signal.shape[1])
    y = np.arange(signal.shape[0])
    r, c = np.meshgrid(x, y)
    I = signal
    surf = ax.plot_surface(c, r, I, cmap = plt.cm.brg)
    if isinstance(zlim, int): 
        ax.set_zlim(0,zlim)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    return fig.tight_layout

def threshold_mad(im: np.ndarray, k=1.5):
    med = np.median(im)
    mad = np.median(np.abs(im.astype(np.float32) - med))
    return med + mad * k * 1.4826

def showImgProfile_old(Image, yLine = None, orient = 0, lw = 1, eqHist = 1, ymin = None, ymax = None, cMap = 'gray'):
    Image_eq = exposure.equalize_hist(Image) if eqHist==1 else Image
    valid = {0, 1,}
    if orient not in valid:
        raise ValueError("results: orient must be one of %r." % valid)
    startPt = [int(0), int(0)]
    endPt = [int(Image.shape[0]), int(Image.shape[1])]
    if yLine != None and yLine > Image.shape[orient]:
        raise ValueError(f'Enter a valid yLine parameter. It has to be a number and less than {Image.shape[orient]}')
    else:
        startPt[orient] = int(Image.shape[orient]//2) if not yLine else int(yLine)
        endPt[orient] = int(Image.shape[orient]//2) if not yLine else int(yLine)        
    row, col = draw.line(startPt[0], startPt[1], endPt[0], endPt[1])
    fig, axs = plt.subplots(1, 2, figsize=(12,4), width_ratios=(1,1.5))
    axs[0].imshow(Image_eq, cmap=cMap); 
    axs[0].set_title(getVarName(Image))
    axs[0].plot(col, row, "--r")
    axs[1].plot(measure.profile_line(Image, startPt, endPt, linewidth = lw))
    axs[1].axis(ymin=ymin,ymax=ymax)
    #axs[1].axvline(x=420, color='r')
    return fig.tight_layout

def showImgProfile(Image, yLine = None, orient = 0, lw = 1, eqHist = 1, ymin = None, ymax = None, cMap = 0):
    if eqHist == 1:
        Image_eq = exposure.equalize_hist(Image) 
    elif eqHist == 2:
        Image_eq = exposure.equalize_adapthist(Image)
    else:
        Image_eq = Image
    # Image_eq = exposure.equalize_adapthist(Image) if eqHist==2 else Image
    cmap_col = ['gray', 'viridis', 'plasma', 'spring', 'summer', 'autumn', 'winter', 'cool', 'wistia', 'rainbow', 'jet', 'turbo']
    valid = {0, 1,}
    startPt = [int(0), int(0)]
    endPt = [int(Image.shape[0]), int(Image.shape[1])]
    if cMap not in range(12):
        raise ValueError(f'Enter a valid cMap parameter. It has to be a number and between 0 and {len(cmap_col)-1}')
    else: pass
    if orient not in valid:
        raise ValueError("results: orient must be one of %r." % valid)
    else: pass
    if yLine != None and yLine > Image.shape[orient]:
        raise ValueError(f'Enter a valid yLine parameter. It has to be a number and less than {Image.shape[orient]}')
    else:
        startPt[orient] = int(Image.shape[orient]//2) if not yLine else int(yLine)
        endPt[orient] = int(Image.shape[orient]//2) if not yLine else int(yLine)        
    row, col = draw.line(startPt[0], startPt[1], endPt[0], endPt[1])
    fig, axs = plt.subplots(1, 2, figsize=(12,4), width_ratios=(1,1.2))
    pic = axs[0].imshow(Image_eq, cmap=cmap_col[cMap]); 
    axs[0].set_title(getVarName(Image))
    fig.colorbar(pic, ax=axs[0], shrink= 0.7, fraction = 0.1)
    axs[0].plot(col, row, "--r")
    axs[1].plot(measure.profile_line(Image, startPt, endPt, linewidth = lw))
    axs[1].set_box_aspect(0.6)
    axs[1].axis(ymin=ymin,ymax=ymax)
    #axs[1].axvline(x=420, color='r')
    return fig.tight_layout

def rmCoeffLvl (coe, lvlrm, rmApprox = 0):
    if type(lvlrm) == list:
        c_arr, slcs = pywt.coeffs_to_array(coe)
        if rmApprox == 1: c_arr[slcs[0]] = 0
        for i in lvlrm:
            for j in ['ad','da','dd']: c_arr[slcs[i][j]] = 0
        # c_mod = pywt.array_to_coeffs(c_arr, slcs, output_format='wavedec2')
    else:
        raise ValueError(f'number of levels to remove has to be a list object')
    return pywt.array_to_coeffs(c_arr, slcs, output_format='wavedec2')

def denoiseImageOld (Image, wL = 'db6'):
    psnr = 0
    lvl = pywt.dwtn_max_level(Image.shape, wL)
    coeff = pywt.wavedec2(Image, wL, level = lvl)
    levels = list(range(1, lvl+1, 1))
    for i in levels:
        rmLvl = levels[-i:] # Selects last i elements of the list
        denoiseCoeff = rmCoeffLvl (coeff, rmLvl)
        imRecon = pywt.waverec2(denoiseCoeff, wL, mode = 'symmetric', axes = (-2, -1))
        imRecon = imRecon.astype(np.uint16)
        newPsnr = metrics.peak_signal_noise_ratio(imRecon, Image)
        if newPsnr > psnr: 
            psnr = newPsnr
            imReconFinal = imRecon
        else: break 
    return imReconFinal

def autocorrelation2D (matrix):
    # matrix = matrix - np.mean(matrix) # this step is optional but common in autocorrelation
    f_matrix = np.fft.fft2(matrix) # 2D fft of the matrix
    power_spectrum = f_matrix * np.conj(f_matrix) # fft times its complex conjugate
    autocorr_matrix = np.fft.ifft2(power_spectrum) # inverse fft of power spectrum (autocorrelation in frequency domain
    autocorr_matrix = np.real(autocorr_matrix) # take only real part
    autocorr_matrix = np.fft.fftshift(autocorr_matrix) # shifts the autocorrelation peak to the center
    autocorr_matrix /= autocorr_matrix.max() # Normalize
    return autocorr_matrix

def denoiseImageDebug (Image, wL = 'db6'):
    # psnr = 0
    lvls = np.empty((0))
    Vars = np.empty((0))
    imRecStack = []
    lvl = pywt.dwtn_max_level(Image.shape, wL)
    coeff = pywt.wavedec2(Image, wL, level = lvl)
    levels = list(range(1, lvl+1, 1))
    for i in levels:
        rmLvl = levels[-i:]
        denoiseCoeff = rmCoeffLvl (coeff, rmLvl)
        imRecon = pywt.waverec2(denoiseCoeff, wL, mode = 'symmetric', axes = (-2, -1))
        imRecon = imRecon.astype(np.uint16)
        imReconAc = autocorrelation2D(Image.astype(np.int32) - imRecon)
        lvls = np.append(lvls, len(rmLvl))
        Vars = np.append(Vars, np.var(imReconAc.flatten()))
        # newPsnr = metrics.peak_signal_noise_ratio(imRecon, Image)
        imRecStack.append(imRecon)
    # create a new figure
    fig = plt.figure()
    # plot to it
    plt.plot(lvls, Vars, marker = 'o', figure=fig) 
    plt.plot(lvls, Vars, marker = 'o', figure=fig)
    plt.xlabel("# of decomposed levels removed", figure=fig)
    plt.ylabel("Variance of Noise removed", figure=fig)
    plt.yscale("log")
    return np.array(imRecStack), Vars, lvls, fig

def denoiseImage (Image, wL = 'db6'):
    '''
    Revised on 9/28/24: The for loop breaks based on variance in autocorrelation instead of PSNR 
    '''
    Var = 1
    lvl = pywt.dwtn_max_level(Image.shape, wL)
    coeff = pywt.wavedec2(Image, wL, level = lvl)
    levels = list(range(1, lvl+1, 1))
    for i in levels:
        rmLvl = levels[-i:]
        denoiseCoeff = rmCoeffLvl (coeff, rmLvl)
        imRecon = pywt.waverec2(denoiseCoeff, wL, mode = 'symmetric', axes = (-2, -1))
        imRecon = imRecon.astype(np.uint16)
        imReconAc = autocorrelation2D(Image.astype(np.int32) - imRecon)
        newVar = np.var(imReconAc.flatten())
        # newPsnr = metrics.peak_signal_noise_ratio(imRecon, Image)
        if Var/newVar > 1: 
            Var = newVar
            imReconFinal = imRecon
        else: break
    return imReconFinal

def denoImg (Image, lvlrm, wL = 'db6'):
    """
    Initial code written on 10/04/24 
    """
    lvl = pywt.dwtn_max_level(Image.shape, wL)
    coeff = pywt.wavedec2(Image, wL, level = lvl)
    levels = list(range(1, lvl+1, 1))
    if lvlrm <= lvl: 
        rmLvl = levels[-lvlrm:]
    else:
        raise ValueError(f'lvlrm has to be a number and less than or equal to {lvl}')
    denoiseCoeff = rmCoeffLvl (coeff, rmLvl)
    imRecon = pywt.waverec2(denoiseCoeff, wL, mode = 'symmetric', axes = (-2, -1))
    return imRecon.astype(np.uint16)

def baseline_als(y, lam, p, niter=9):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = sparse.linalg.spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def find_membrane(sigImage):
    fact = 1
    mem = sigImage > (np.median(sigImage) * fact) # this is calulated just to get the while loop started
    # memArea, numRegs = measure.label(mem, return_num=True)
    # memProp = measure.regionprops_table(memArea, properties = ('label', 'area'))
    # mem[~(memArea == memProp['label'][memProp['area'] == max(memProp['area'])])] = False
    # mem = ndimage.binary_fill_holes(mem)
    while np.sum(mem) / (mem.shape[0] * mem.shape[1]) < 0.78:
        fact = fact - 0.1
        mem = sigImage > (np.median(sigImage) * fact)
        memArea, numRegs = measure.label(mem, return_num=True)
        memProp = measure.regionprops_table(memArea, properties = ('label', 'area'))
        mem[~(memArea == memProp['label'][memProp['area'] == max(memProp['area'])])] = False
        mem = ndimage.binary_fill_holes(mem)
        # new_fact = fact - 0.1
    mem = morphology.binary_dilation(mem, footprint = [(np.ones((50, 1)), 1), 
                                                      (np.ones((1, 50)), 1)])
    mem = morphology.binary_erosion(mem, footprint = [(np.ones((200, 1)), 1), 
                                                      (np.ones((1, 200)), 1)])
    return mem, np.sum(mem) / (mem.shape[0] * mem.shape[1])

def find_edge(sig1D, ht, plot = False):
    """
    Input:
        1. 1D signal
        2. plot output: true or false
    Output: 
        1. location of edge on the left
        2. location of edge on the right
        3. if plot == True; plot of gradient of 1d signal with peak locations
    """
    sig1D_G = ndimage.gaussian_filter1d(np.gradient(sig1D), sigma=10)
    peaksL, propL = signal.find_peaks(sig1D_G, height=ht, width = 0, rel_height=0.9) #height was 50
    peaksR, propR = signal.find_peaks(-sig1D_G, height=ht, width = 0, rel_height=0.9) #height was 50
    if plot == True:
        fig = plt.figure()
        plt.plot(sig1D_G, figure = fig)
        plt.plot(peaksL, sig1D_G[peaksL], "x", figure = fig)
        plt.plot(peaksR, sig1D_G[peaksR], "o", figure = fig)
        return peaksL[0] + (propL['widths'][0]/2), peaksR[-1] - (propR['widths'][-1]/2), fig
    else: 
        return peaksL[0] + (propL['widths'][0]/2), peaksR[-1] - (propR['widths'][-1]/2)

def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    """
    This is based on a new paper: "Baseline correction using asymmetrically reweighted penalized least squares smoothing" by
    Sung-June Baek,a Aaron Park,*a Young-Jin Ahna and Jaebum Choo*b
    """
    L = len(y)
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    crit = 1
    count = 0
    while crit > ratio:
        z = sparse.linalg.spsolve(W + H, W * y) # Last change is in this line: "sparse."
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
        crit = norm(w_new - w) / norm(w)
        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1
        if count > niter:
            # print('Maximum number of iterations exceeded')
            break
    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z

def baseline_als_optimized(y, lam, p, niter=10):
    """
    Changes from the function baseline_als2: 
    - lam*D.dot(D.transpose()) is calculated outside of the for loop, since it doesn't change
    - instead of creating new matrix in every for loop, just update the changed diagonal values 
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baseline_corrected_img(Image, ImageBW, full_output = False, method = 0):
    im_sig_corrX = np.zeros_like(Image)
    im_sig_corrY = np.zeros_like(Image)
    # mlsigx = np.zeros_like(Image)
    # mlsigy = np.zeros_like(Image)
    for i in range(0,Image.shape[0]):
        sig1dBL = np.copy(Image[i,:])
        sig1dBL2 = np.copy(sig1dBL[ImageBW[i,:]])
        if sig1dBL2.size <= 3:
            sig1d_corr = np.zeros_like(sig1dBL)
        else:
            if method == 1:
                sig1d_bl = baseline_arPLS(sig1dBL2, ratio=0.00001, lam=1e7, niter=10, full_output=False)
            else:
                sig1d_bl = baseline_als_optimized(sig1dBL2, 100000, 0.0001) 
            sig1dBL[ImageBW[i,:]] = sig1d_bl
            sig1d_corr = Image[i,:].astype(np.int32) - sig1dBL
            sig1d_corr[sig1d_corr < 0] = 0
            # peaks, properties = signal.find_peaks(sig1d_corr, rel_height=0.9, width=70, distance = 200, prominence=0)
            # mlsigx[i,:][peaks] = 1
        im_sig_corrX[i,:] = sig1d_corr
    im_sig_corrX = np.asarray(im_sig_corrX, dtype=np.uint16)
    
    for i in range(0,Image.shape[1]):
        sig1dBL = np.copy(Image[:,i])
        sig1dBL2 = np.copy(sig1dBL[ImageBW[:,i]])
        if sig1dBL2.size <= 3:
            sig1d_corr = np.zeros_like(sig1dBL)
        else:
            if method == 1:
                sig1d_bl = baseline_arPLS(sig1dBL2, ratio=0.00001, lam=1e7, niter=10, full_output=False)
            else:
                sig1d_bl = baseline_als_optimized(sig1dBL2, 100000, 0.0001)
            sig1dBL[ImageBW[:,i]] = sig1d_bl
            sig1d_corr = Image[:,i].astype(np.int32) - sig1dBL
            sig1d_corr[sig1d_corr < 0] = 0
            # peaks, properties = signal.find_peaks(sig1d_corr, rel_height=0.9, width=70, distance = 200, prominence=0)
            # mlsigy[:,i][peaks] = 1
        im_sig_corrY[:,i] = sig1d_corr
    im_sig_corrY = np.asarray(im_sig_corrY, dtype=np.uint16)
    im_sig_corr = np.minimum(filters.median(im_sig_corrX, morphology.disk(4)), filters.median(im_sig_corrY,morphology.disk(4)))
    return im_sig_corr

def baseline_als2(y, lam, p, niter=10):
    """
    Changes from the function baseline_als: 
    - diags is used instead of csc_matrix and np.diff. This should speed up the processing
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z
