from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import cv2
import numpy as np
import sys

"""
This file converts a Dataset of images
into a Dataset of 64x64 jpg images ready
to be used for WGAN training
"""

# Defined paths and counter
out_path = "out/"
in_path = "train/"
counter = 1

# The function that reformats all images
print("Converting...")
for im_path in os.listdir(in_path):
    if ".png" not in im_path: continue
    img = Image.open(in_path+im_path)
    im = img.resize((64, 64), Image.BILINEAR)

    im = Image.fromarray(np.uint8(np.where(np.array(im) > 0, 1, 0)*255))
    im.save(out_path+str(counter)+".jpg")
    counter += 1
print("Finished!")


