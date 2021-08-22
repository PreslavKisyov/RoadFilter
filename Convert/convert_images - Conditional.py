from PIL import Image
import os
import numpy as np
import sys
import Augmentor

"""
This file converts a dataset of images into different size and labels
used for preprocessing a Conditional WGAN Dataset
"""

# Defined variables with paths
out_path = "out/"
in_path = "train/"
labels = ["_small", "_big"]
img_format = ".png"  # might need to be changed depending on the dataset

# This method converts every image from a folder
# into a 64x64 Binary Image and assigns a label
# given the percentage of road in it.
# @param in_path The path to the images
# @param out_path The path to the output folder
# @param labaels The labels for the images
# @param pct The percentage that decides the labels
def convert_data(in_path, out_path, labels, pct=35):
    counter = 1
    print("Converting...")
    for im_path in os.listdir(in_path):
        if img_format not in im_path: continue
        # os.rename(in_path+im_path, str(counter)+".jpg")  # if renaming is needed
        img = Image.open(in_path+im_path)
        np.set_printoptions(threshold=sys.maxsize)
        img = np.array(img.resize((64, 64), Image.BILINEAR))
        count = np.count_nonzero(img)
        pct_count = (count/float(img.shape[0]*img.shape[1])) * 100.0  # Get percentage of road
        img = Image.fromarray(np.uint8(np.where(np.array(img) > 0, 1, 0)*255))
        if pct_count > pct: img.save(out_path + str(counter) + labels[1] + ".jpg")
        else: img.save(out_path + str(counter) + labels[0] + ".jpg")
        counter += 1
    print("Finished!")

# This method augments the data
# by rotating and zooming
# @param path The path to the images
def augment_data(path):
    print("Augmenting...")
    augmentor = Augmentor.Pipeline(path)
    augmentor.rotate90(1)
    augmentor.process()
    augmentor.rotate270(1)
    augmentor.process()
    print("Finished!")

# 1st) Augment_data
# augment_data(in_path[:-1])  # Remove last \
# 2nd) Convert data
# Do only after moving output folder to main folder after augmentation
convert_data(in_path, out_path, labels)