import keras
import numpy as np
from PIL import Image
import cv2.cv2 as cv2

import utilityFunctions as uf

class Roads:

    # A static variable that defines the path for the model
    path = "stock-filters/Roads/RoadWGAN.h5"

    def __init__(self, n_maps=2, n_imgs=25):
        self.latent_dim = 100
        self.n_imgs = n_imgs
        self.n_maps = n_maps

        # Execute functions
        imgs = self.load_model()
        self.road_network = self.get_road_network(imgs)

    def load_model(self):
        model = keras.models.load_model(self.path)
        noise = np.random.random(self.n_imgs * self.latent_dim).reshape(self.n_imgs, self.latent_dim)
        # Generate n_imgs number of images
        imgs = model.predict(noise)
        imgs = (imgs + 1) / 2.0  # Normalize to between 0-1

        return imgs

    # From n_imgs images, perform Preprocessing
    # and extract one road network image
    def get_road_network(self, imgs):
        road_maps = list()
        image_index = 0

        while len(road_maps) != self.n_maps:
            # If there are no more valid road maps (above the pct)
            if image_index > self.n_imgs:
                index_list = np.random.random_integers(self.n_imgs, size=self.n_maps - len(road_maps))
                for i in index_list: road_maps.append(imgs[i, :, :, 0])
                break

            # Get the percentage of roads present
            img = imgs[image_index, :, :, 0]
            count = np.count_nonzero(img)
            pct_count = (count/float(img.shape[0]*img.shape[1])) * 100.0
            image_index += 1
            # Add a road if it contains at least 20% of road infrastructure
            if pct_count > 20.0: road_maps.append(img)

        # Combine all road maps into one
        road_network = road_maps[0]
        for road_map_index in range(1, self.n_maps):
            road_network += road_maps[road_map_index]

        return road_network

    def process_road_network(self, region_size):
        # Get the required size and convert to grayscale image
        size = np.mgrid[region_size.minx:region_size.maxx, region_size.minz:region_size.maxz][0].shape
        min_size, max_size = min(size), max(size)
        road_network_gray = np.array(self.road_network * 255, dtype=np.uint8)

        # Perform Closing (Dilation -> Erosion)
        kernel = np.ones((2, 2), np.uint8)
        road_network_dilated = cv2.dilate(road_network_gray, kernel)
        road_network_closed = cv2.erode(road_network_dilated, kernel)

        # Remove Disconnected Components
        components, objects, stats, _ = cv2.connectedComponentsWithStats(road_network_closed, connectivity=4)
        sizes = stats[1:, -1]  # Remove the background as a component
        max_component = np.where(sizes == max(sizes))[0] + 1
        road_network_closed[objects != max_component] = 0

        # Resize to the required size using Interpolation and convert to Binary
        processed_road_network = np.array(Image.fromarray(road_network_closed).resize((min_size, min_size), Image.LANCZOS))
        return cv2.threshold(processed_road_network, int(np.mean(processed_road_network)), 255, cv2.THRESH_BINARY)[1]


"""
This is the main function that 
gets called after executing the filter in MCEdit
"""
def perform(level, box, options):
    road_network = Roads().process_road_network(box)  # Get the road as a Numpy array Image
    biom, road_blocks, log_types = get_biom(level, box)  # Get the biom and types of blocks
    floor = get_floor(level, box, road_network)
    build_road(level, floor, road_network)

def build_road(level, floor_points, road):
    for pos in floor_points:
        x, y, z = floor_points[pos]

        # # TODO: check the elevetation level and record the road position
        if pos in tuple(zip(*np.where(road == 255))):
            # Build the road here
            uf.setBlock(level, (49, 0), x, y, z)
            # Remove trees if on road
            remove_tree(level, x, y, z)

def remove_tree(level, x, y, z):
    points_x, points_y, points_z = [], [y], []
    y += 1

    while True:
        if level.blockAt(x, y, z) in [17, 18, 81, 161, 162]: # Removes all trees TODO: Move it so you remove only if on path way
            points_y.append(y)
            points_x = get_x_bound(level, x, y, z, points_x)
            points_z = get_z_bound(level, x, y, z, points_z)
            y += 1
        else: break
    remove_tree_from_bound(level, points_x, points_y, points_z)

def remove_tree_from_bound(level, points_x, points_y, points_z):
    if points_x and points_y and points_z:
        minx, maxx = min(points_x), max(points_x)
        miny, maxy = min(points_y), max(points_y)
        minz, maxz = min(points_z), max(points_z)

        for x in range(minx-1, maxx+1):
            for z in range(minz-1, maxz+1):
                for y in range(miny-1, maxy+1):
                    if level.blockAt(x, y, z) in [17, 18, 81, 161, 162]: uf.setBlock(level, (0, 0), x, y, z)

def get_x_bound(level, x, y, z, points_x):
    posx =  x + 1
    reverse = False
    while True:
        if reverse is False:
            if level.blockAt(posx, y, z) in [17, 18, 81, 161, 162]: # Removes all trees TODO: Move it so you remove only if on path way
                points_x.append(posx)
                posx += 1
            else:
                posx = x-1
                reverse = True
        else:
            if level.blockAt(posx, y, z) in [17, 18, 81, 161, 162]: # Removes all trees TODO: Move it so you remove only if on path way
                points_x.append(posx)
                posx -= 1
            else: break
    # print("X: ", points_x)
    return points_x

def get_z_bound(level, x, y, z, points_z):
    posz = z + 1
    reverse = False
    while True:
        if reverse is False:
            if level.blockAt(x, y, posz) in [17, 18, 81, 161, 162]: # Removes all trees TODO: Move it so you remove only if on path way
                points_z.append(posz)
                posz += 1
            else:
                posz = z-1
                reverse = True
        else:
            if level.blockAt(x, y, posz) in [17, 18, 81, 161, 162]: # Removes all trees TODO: Move it so you remove only if on path way
                points_z.append(posz)
                posz -= 1
            else: break
    # print("Z: ", points_z)
    return points_z

def get_floor(level, box, road_network):
    mapped_points = {}
    pos_x, pos_y = 0, 0
    for x in range(box.minx, box.maxx): # depth
        for z in range(box.minz, box.maxz): # width
            for y in range(box.maxy, box.miny-1, -1): # height (col) but count the selected level
                #
                # if level.blockAt(x, y, z) in [17, 18, 81, 161, 162]: # Removes all trees TODO: Move it so you remove only if on path way
                #     uf.setBlock(level, (0, 0), x, y, z)
                if level.blockAt(x, y, z) in [1, 2, 3, 12, 13] and (pos_x, pos_y) not in mapped_points.keys():
                    mapped_points[(pos_x, pos_y)] = (x, y, z)
                    break
            pos_y += 1
        pos_y = 0
        pos_x += 1

    return mapped_points

def get_biom(level, box):
    log_types, road_choices = [], []
    # Choice per slice
    for (chunk, slices, point) in level.getChunkSlices(box):
        bins = np.bincount(chunk.root_tag["Level"]["Biomes"].value)
        count = bins/float(len(chunk.root_tag["Level"]["Biomes"].value))
        bioms = np.flatnonzero(count)
        probs = {i:count[i] for i in bioms}
        road_choices.append(np.random.choice(probs.keys(), p=probs.values()))
        log_types = get_log_types(chunk, log_types)

    road_biom = max(road_choices, key=road_choices.count)
    road_blocks = get_road_blocks(road_biom)
    return road_biom, road_blocks, log_types

# TODO: link/cite
# From official Minecraft Wiki (Biomes and Blocks ID)
def get_road_blocks(road_biom):
    # Block, Slab, Stairs
    if road_biom in [2, 7, 16, 17, 27, 28, 36, 37, 38, 39, 130, 165, 166, 167]: return [43, 44, 109]
    else: return [1, 44, 67]

# TODO: link/cite
# Method from the GDMC Competition
# Checks the biom of each chunk and adds a block type
def get_log_types(chunk, log_types):
    for val in chunk.root_tag["Level"]["Biomes"].value:
        if val in [1,3,4,6,8,132]: #oak
            log_types.append("oak")
        if val in [21,22,23,149,151]: #jungle
            log_types.append("jungle")
        if val in [5,12,13,19,30,31,32,33,158,160,161,133]: #spuce
            log_types.append("spruce")
        if val in [6,29,157,134]: #dark
            log_types.append("dark")
        if val in [2,17,27,28,155,156]: #birch
            log_types.append("birch")
        if val in [35,36,37,38,39,163,164,165,166,167]: #acacia
            log_types.append("acacia")

    if not log_types: return ["oak"]
    return log_types