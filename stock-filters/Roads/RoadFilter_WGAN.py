import keras
import numpy as np
from PIL import Image
import cv2.cv2 as cv2
import A_star as star
import utilityFunctions as uf
from itertools import combinations

"""
    This class creates and processes the Road Network Image.
    A WGAN Generates a road image and then it is adapted
    to the selected terrain using Computer Vision techniques.
    
    @author: Preslav Kisyov
"""
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
                for i in index_list: road_maps.append(imgs[idw, :, :, 0])
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
    floor, usable_floor = get_floor(level, box)  # Get the usable floor box region
    build_road(level, floor, road_network, box, usable_floor) # Build and connect roads


def build_road(level, floor_points, road, box, usable_floor):
    # road_size = np.array(np.where(road == 255)).shape[-1]

    for pos in sorted(floor_points):
        x, y, z = floor_points[pos]

        # Remove Lava
        remove_lava(level, x, y, z, box)
        # TODO: check the elevation level and find obstacles

        if pos in tuple(zip(*np.where(road == 255))) and level.blockAt(x, y, z) not in [0, 8, 9]:
            # Remove trees if on road
            remove_tree(level, x, y, z)
            # TODO: Build the road here
            uf.setBlock(level, (98,0), x, y, z)
            usable_floor[pos[0]][pos[1]] = 255

    # Get Components
    components, usable_floor = get_components(level, usable_floor, floor_points)
    floor = np.zeros(usable_floor.shape) if usable_floor is not None else None
    # Check if there are separated components
    if components is not None:
        connect_components(level, components, floor, floor_points, np.array(usable_floor))

def connect_components(level, components, floor, floor_points, usable_floor):
    start_end_points = list(combinations(components.keys(), 2))
    print("There are " + str(len(components)) + " disconnected roads!")
    visited = set()
    for points in start_end_points:
        if points[0] in visited: continue
        visited.add(points[0])
        path = star.search(floor, 1, components[points[0]], components[points[1]])
        ar_path = np.array(path)

        if path is not None:
            rows, cols = np.where(ar_path != -1)
            for i in range(len(rows)-1): usable_floor[rows[i]][cols[i]] == 2
            connect_roads(rows, cols, level, floor_points, usable_floor)

def connect_roads(rows, cols, level, floor_points, usable_floor):
    block = (98, 0)
    is_start, is_end = False, False
    start, end = [rows[0], cols[0]], [rows[-1], cols[-1]]
    for i in range(len(rows)-1):
        r, c = rows[i], cols[i]
        x, y, z = floor_points[(r, c)]
        # TODO: Build road or bridge
        # TODO: remove unnecessary path
        if level.blockAt(x, y, z) in [8, 9]:  # If path is on water block
            uf.setBlock(level, block, x, y+1, z)
            if not is_start: build_bridge(level, block, x, y, z)
            else: is_start = False
            is_end = True
        else:
            # TODO: adds planks even not on proper places
            if is_end:
                uf.setBlock(level, (44, 0), x, y+1, z)
                is_end = False
            uf.setBlock(level, block, x, y, z)  # TODO: Change to actual path 98
            if usable_floor[r+1][c+1] != len(usable_floor) and usable_floor[r+1][c+1] == 1:
                uf.setBlock(level, (44, 0), x, y+1, z)  # Put step at the start
                is_start = True

def build_bridge(level, block, x, y, z):
    # Build walls
    wall_points = [(x+1, y+1, z+1), (x-1, y+1, z-1), (x+1, y+1, z-1), (x-1, y+1, z+1), (x, y+1, z+1), (x, y+1, z-1), (x+1, y+1, z), (x-1, y+1, z)]
    for p in wall_points:
        if level.blockAt(p[0], p[1], p[2]) not in [block[0], 8, 9] and level.blockAt(p[0], p[1]-1, p[2]) in [8, 9]: uf.setBlock(level, block, p[0], y+2, p[2])
    if level.blockAt(x, y+2, z) in [block[0]]: uf.setBlock(level, (0, 0), x, y+2, z)

def get_components(level, usable_floor, floor_points):
    try:
        ar_floor = np.array(usable_floor, dtype=np.uint8 )
    except:
        print("Invalid floor has been selected!\nNo components will be found!")
        raise
        return None, None

    components, objects, stats, _ = cv2.connectedComponentsWithStats(ar_floor, connectivity=8)

    if components < 3: return None, None # The background + 2 separated roads
    comp = np.arange(1, components) # all components except the background
    components_map = {}
    for x in range(0, ar_floor.shape[0]):
        for y in range(0, ar_floor.shape[1]):
            posx, posy, posz = floor_points[(x, y)]
            if level.blockAt(posx, posy, posz) in [8, 9]: ar_floor[x][y] = 1

            if objects[x][y] in comp:  # Record the positions of each component
                components_map[objects[x][y]] = [x, y]
                comp = np.delete(comp, np.argwhere(comp == objects[x][y]))
    return components_map, ar_floor

# This method removes any detected lava
def remove_lava(level, x, y, z, box):
    block = level.blockAt(x, y, z)
    c_x, c_y = 0, 0
    # Get the closest non-obstacle block
    while block in [8, 9, 10, 11]:
        if x+c_x != box.maxx:
            block = level.blockAt(x+c_x, y, z)
            c_x += 1
        else:
            block = level.blockAt(x, y+c_y, z)
            c_y += 1

        if y+c_y == box.maxy: block = 0

    # If there is lava
    if level.blockAt(x, y, z) in [10, 11]: uf.setBlock(level, (block, 0), x, y, z)
    elif level.blockAt(x, y+1, z) in [10, 11]: uf.setBlock(level, (block, 0), x, y+1, z)

def remove_tree(level, x, y, z):
    points_x, points_y, points_z = [], [y], []
    y += 1

    while True:
        if level.blockAt(x, y, z) in [17, 18, 81, 161, 162]: # Removes all trees
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
            if level.blockAt(posx, y, z) in [17, 18, 81, 161, 162]:
                points_x.append(posx)
                posx += 1
            else:
                posx = x-1
                reverse = True
        else:
            if level.blockAt(posx, y, z) in [17, 18, 81, 161, 162]:
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
    return points_z

def get_floor(level, box):
    mapped_points = {}
    usable_floor = []
    pos_x, pos_y = 0, 0
    for x in range(box.minx, box.maxx): # depth
        col = []
        for z in range(box.minz, box.maxz): # width
            for y in range(box.maxy, box.miny-1, -1): # height (col) but count the selected level # in [1, 2, 3, 8, 9, 10, 11, 12, 13, 78, 80]
                if level.blockAt(x, y, z) in [1, 2, 3, 8, 9, 10, 11, 12, 13, 78, 80] and (pos_x, pos_y) not in mapped_points.keys():
                    mapped_points[(pos_x, pos_y)] = (x, y, z)
                    col.append(0)
                    break
                elif level.blockAt(x, y-1, z) in [1, 2, 3, 8, 9, 10, 11, 12, 13, 78, 80] and (pos_x, pos_y) not in mapped_points.keys():
                    mapped_points[(pos_x, pos_y)] = (x, y-1, z)
                    col.append(0)
                    break
            pos_y += 1
        usable_floor.append(col)
        pos_y = 0
        pos_x += 1

    return mapped_points, usable_floor

def get_biom(level, box):
    log_types, road_choices = [], []
    # Choice per slice
    for (chunk, slices, point) in level.getChunkSlices(box):
        bins = np.bincount(chunk.root_tag["Level"]["Biomes"].value)
        count = bins/float(len(chunk.root_tag["Level"]["Biomes"].value))
        bioms = np.flatnonzero(count)
        probs = {i:count[i] for i in bioms}
        road_choices.append(np.random.choice(probs.keys(), p=probs.values()))
        # log_types = get_log_types(chunk, log_types)

    road_biom = max(road_choices, key=road_choices.count)
    road_blocks = get_road_blocks(road_biom)
    return road_biom, road_blocks, log_types

# TODO: link/cite
# From official Minecraft Wiki (Biomes and Blocks ID)
def get_road_blocks(road_biom):
    # Block, Slab, Stairs
    if road_biom in [2, 7, 16, 17, 27, 28, 36, 37, 38, 39, 130, 165, 166, 167]: return [43, 44, 109]
    else: return [1, 44, 67]

# # TODO: link/cite
# # Method from the GDMC Competition
# # Checks the biom of each chunk and adds a block type
# def get_log_types(chunk, log_types):
#     for val in chunk.root_tag["Level"]["Biomes"].value:
#         if val in [1,3,4,6,8,132]: #oak
#             log_types.append("oak")
#         if val in [21,22,23,149,151]: #jungle
#             log_types.append("jungle")
#         if val in [5,12,13,19,30,31,32,33,158,160,161,133]: #spuce
#             log_types.append("spruce")
#         if val in [6,29,157,134]: #dark
#             log_types.append("dark")
#         if val in [2,17,27,28,155,156]: #birch
#             log_types.append("birch")
#         if val in [35,36,37,38,39,163,164,165,166,167]: #acacia
#             log_types.append("acacia")
#
#     if not log_types: return ["oak"]
#     return log_types