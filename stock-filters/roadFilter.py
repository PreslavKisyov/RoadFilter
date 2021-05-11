from __future__ import division
import time # for timing
from math import sqrt, tan, sin, cos, pi, ceil, floor, acos, atan, asin, degrees, radians, log, atan2, acos, asin
from random import *
from numpy import *
import utilityFunctions as uf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# TODO: Find the selected box size/shape
# TODO: Connect the edges between the points on the rescaled road image
# TODO: Do not count trees in or other obstacles
def get_maps(level, box):
    mapped_points = {}
    pos_x = 0
    pos_y = 0
    for x in range(box.minx, box.maxx): # depth
        for z in range(box.minz, box.maxz): # width
            for y in range(box.maxy, box.miny-1, -1): # height (col) but count the selected level
                if level.blockAt(x, y, z) in [17, 18, 81, 161, 162]: # Removes all trees TODO: Move it so you remove only if on path way
                    uf.setBlock(level, (0, 0), x, y, z)

                if level.blockAt(x, y, z) in [1, 2, 3, 12, 13] and (pos_x, pos_y) not in mapped_points.keys():
                    mapped_points[(pos_x, pos_y)] = (x, y, z)
                    break
                    # uf.setBlock(level, (49, 0), x, y, z)
            pos_y += 1
        pos_y = 0
        pos_x += 1

    print(len(mapped_points))
    im = cv2.imread('C:\\Users\\Preslav\\Desktop\\Thesis\\Thesis\\Software\\Road Networks\\0.jpg')

    im = cv2.resize(im, (100, 100))
    plt.imshow(im)
    plt.show()

    # Builds
    for pos in mapped_points:
        # print(pos)
        if pos in tuple(zip(*np.where(im != 255)[:-1])):
            x, y, z = mapped_points[pos]
            uf.setBlock(level, (49, 0), x, y, z)


# This function is the main one called by the filter
def perform(level, box, options):
    print "This is the level: " + str(level)
    print "This is the box: " + str(box)
    bottom = get_maps(level, box)
    print "This is the bottom: " + str(bottom)
