import tkinter, glob
from tkinter import filedialog, END
from matplotlib import pyplot as plt
from skimage import io, measure, filters, morphology
import numpy as np
import os
import pandas as pd
import datetime

import helpers

plt.ioff()

def process_images(run_dir):
  sig_imgs = []
  sig_imgs_names = []
  analysis_log = pd.DataFrame()
  properties = ('label', 'area', 'area_bbox', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'bbox',
                'centroid', 'centroid_local', 'centroid_weighted', 'centroid_weighted_local', 'coords_scaled', 'coords', 'eccentricity',
                'equivalent_diameter_area', 'euler_number', 'extent', 'feret_diameter_max', 'image', 'image_convex', 'image_filled',
                'image_intensity', 'inertia_tensor', 'inertia_tensor_eigvals', 'intensity_max', 'intensity_mean', 'intensity_min',
                'intensity_std', 'moments', 'moments_central', 'moments_hu', 'moments_normalized', 'moments_weighted', 'moments_weighted_central',
                'moments_weighted_hu', 'moments_weighted_normalized', 'num_pixels', 'orientation', 'perimeter', 'perimeter_crofton',
                'slice', 'solidity')

  roi_save_path = run_dir + '\\improc\\chetan\\' + 'ROI Image'
  corr_img_profile_save_path = run_dir + '\\improc\\chetan\\' + 'Corrected Image'
  sig_img_path = run_dir + '\\improc\\signal\\*.tif'
  csv_filepath = run_dir + '\\Region_Properties'+ datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv'
  
  tif_files = glob.glob(sig_img_path)
  plt.ioff()
  if os.path.isdir(run_dir + '\\improc\\chetan') == False:
    os.mkdir(run_dir + '\\improc\\chetan')
  
  for file in tif_files:
    signal = io.imread(file, plugin = 'pil')
    
    signal = helpers.denoiseImage(signal)
    membrane, memCoverage = helpers.find_membrane(signal)
    mem_roi = helpers.showRoiBoundary(signal, membrane)
    # io.imsave(roi_save_path + os.path.basename(file) + '.png', mem_roi)
    plt.savefig(roi_save_path + os.path.basename(file) + '1.png') # This works if showRoiBoundary() returns fig.tight_layout
    # mem_roi.savefig(roi_save_path + os.path.basename(file) + '.png') # This works if showRoiBoundary() returns plt.gcf()
    # plt.imsave(roi_save_path + os.path.basename(file), mem_roi)

    corr_signal = helpers.baseline_corrected_img(signal, membrane, method = 1)
    corr_img_profile = helpers.showImgProfile(corr_signal, eqHist = 2)
    plt.savefig(corr_img_profile_save_path + os.path.basename(file) + '.png')
    # io.imsave(corr_img_profile_save_path + os.path.basename(file), corr_img_profile)

    corr_signal = filters.median(corr_signal, footprint = morphology.disk(7))
    sig_img_save_path = corr_img_profile_save_path + 'baseline_corrected' + os.path.basename(file)
    io.imsave(sig_img_save_path, corr_signal.astype(np.uint16))

    thresh_signal = corr_signal > helpers.threshold_mad(corr_signal)
    sig_roi = helpers.showRoiBoundary(corr_signal, thresh_signal)
    plt.savefig(roi_save_path + os.path.basename(file) + '2.png')

    labelled_signal, numRegs = measure.label(thresh_signal, return_num = True)
    region_prop = pd.DataFrame(measure.regionprops_table(labelled_signal, intensity_image = corr_signal, properties = properties))
    if analysis_log.empty:
      analysis_log = region_prop
    else:
      analysis_log = pd.concat([analysis_log, region_prop])

  analysis_log.to_csv(csv_filepath)



# sig_imgs.append(io.imread(file, plugin = 'pil'))
# sig_imgs_names.append(os.path.basename(file))
  
# for i in range(len(sig_imgs)):
#   sig_img_save_path = run_dir + '\\improc\\chetan\\' + 'edited' + sig_imgs_names[i]
#   io.imsave(sig_img_save_path, sig_imgs[i].astype(np.uint16))



def select_dir(entry):
  selected_dir = filedialog.askdirectory()
  if selected_dir is not None and selected_dir != '':
    entry.delete(0, END)
    entry.insert(0, selected_dir)
  return entry



root = tkinter.Tk()
root.title('Image Processing Routine')
root.minsize(500, 800)
root.maxsize(1000, 1600)

w = tkinter.Label(root, text = 'Selected Folder: ')
w.place(x=20, y=10)
Img_dir = tkinter.Entry(root, width=48)
Img_dir.place(x=100, y=10)
sfbtn = tkinter.Button(root, width=2, text="...",
                       command= lambda:select_dir(Img_dir))
sfbtn.place(x=400, y=10)

tkinter.Button(text='Process Images', command=lambda: process_images(Img_dir.get())).place(x= 200, y=50)
'''
Widgets are added here
'''

#w.pack()

root.mainloop()