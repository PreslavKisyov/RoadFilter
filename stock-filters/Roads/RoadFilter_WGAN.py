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
    The road is then placed in Minecraft and further processed
    to resemble a real-like road system.
    
    @author: Preslav Kisyov
"""
class Roads:

    # A static variable that defines the path for the model
    path = "stock-filters/Roads/RoadWGAN.h5"

    # This is the main method of the class
    # It initializes vital variables
    # @param n_maps The number of roads to combine
    # @param n_imgs The number of images generated
    def __init__(self, n_maps=2, n_imgs=25):
        self.latent_dim = 100
        self.n_imgs = n_imgs
        self.n_maps = n_maps
        self.usable_floor = None
        self.floor = None
        self.blocks = None

        # Execute functions
        imgs = self.load_model()
        self.road_network = self.get_road_network(imgs)

    # This method loads a keras model from path
    # and produces noise to be passed through the model
    # @return The predicted WGAN images
    def load_model(self):
        model = keras.models.load_model(self.path)
        noise = np.random.random(self.n_imgs * self.latent_dim).reshape(self.n_imgs, self.latent_dim)
        # Generate n_imgs number of images
        imgs = model.predict(noise)
        imgs = (imgs + 1) / 2.0  # Normalize to between 0-1

        return imgs

    # From n_imgs images, perform Preprocessing
    # and extract one road network image
    # @param imgs The list of images
    # @return road_network One generate road image
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

    # This method processes the road image
    # by applying Computer Vision methods to clean it
    # @param region_size The size of the selected Minecraft region
    # @return A processed road image
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
        processed_road_network = np.array(Image.fromarray(road_network_closed).resize((min_size, max_size), Image.LANCZOS))
        return cv2.threshold(processed_road_network, int(np.mean(processed_road_network)), 255, cv2.THRESH_BINARY)[1]

    # This method build the road if not in water
    # @param level The level provided by MCEdit
    # @param floor_points, usable_floor The whole floor map and the array containing the roads
    # @param road The road generated image as an array
    # @param box The bounding box provided by MCEdit
    # @param blocks A list of block types
    def build_road(self, level, floor_points, road, box, usable_floor, blocks):
        for pos in sorted(floor_points):
            x, y, z = floor_points[pos]

            # Remove Lava
            self.remove_lava(level, x, y, z, box)

            if pos in tuple(zip(*np.where(road == 255))) and level.blockAt(x, y, z) not in [0, 8, 9]:
                # Remove trees if on road
                self.remove_tree(level, x, y, z)
                # Build the road
                uf.setBlock(level, blocks[0], x, y, z)
                # Build tunnel
                self.build_tunnel(level, x, y, z, blocks[0])
                try:
                    usable_floor[pos[0]][pos[1]] = 255
                except: print("Error: Index is out of range for usable_floor!\nNo Roads will be connected!\nProbably the floor is invalid!")

        # Assign floor array and map
        self.usable_floor = usable_floor
        self.floor = floor_points
        self.blocks = blocks

        # Get Components
        components, usable_floor = self.get_components(level, usable_floor, floor_points)
        floor = np.zeros(usable_floor.shape) if usable_floor is not None else None
        # Check if there are separated components
        if components is not None:
            self.connect_components(level, components, floor, floor_points, np.array(usable_floor), blocks)

    # This method builds a tunnel if there are block above the path
    # @param level The level provided by MCEdit
    # @param x, y, z The coordinates of the path
    # @param block The type of block
    def build_tunnel(self, level, x, y, z, block):
        for i in range(1, 3): # Make the tunnel 2 blocks high
            if level.blockAt(x, y+i, z) in [1, 2, 3, 12, 13, 78, 80]:
                uf.setBlock(level, (0, 0), x, y+i, z)
                block_points = [(x+1, z), (x+1, z+1), (x+1, z-1), (x-1, z),
                                (x-1, z+1), (x-1, z-1), (x, z+1), (x, z-1)]
                is_tunnel_roof = False
                for p in block_points:
                    if level.blockAt(p[0], y+3, p[1]) != 0: is_tunnel_roof = True; break
                # Build tunnel ceiling if necessary
                if is_tunnel_roof and level.blockAt(x, y+3, z) in [0, 17, 81, 162, 18, 161]: uf.setBlock(level, block, x, y+3, z)

    # This method tries to find a path between every disconnected road
    # @param level The level provided by MCEdit
    # @param components The map of disconnected roads
    # @param free_floor The floor without any obstructions
    # @param floor_points, usable_floor The whole floor map and the array containing the roads
    # @param blocks A list of block types
    def connect_components(self, level, components, free_floor, floor_points, usable_floor, blocks):
        start_end_points = list(combinations(components.keys(), 2))
        print("There are " + str(len(components)) + " disconnected roads!")
        visited = set()
        # For every disconnected road point
        for points in start_end_points:
            if points[0] in visited: continue
            visited.add(points[0])
            path = star.search(free_floor, 1, components[points[0]], components[points[1]])
            ar_path = np.array(path)

            if path is not None:  # Build connecting road
                rows, cols = np.where(ar_path != -1)
                for i in range(len(rows)-1): usable_floor[rows[i]][cols[i]] == 255  # Add the new path
                self.connect_roads(rows, cols, level, floor_points, usable_floor, blocks)

        # Update floor array
        self.usable_floor = usable_floor

    # This method uses the A* path to connect the disconnected roads
    # @param rows, cols The rows and columns of the path array
    # @param level The level provided by MCEdit
    # @param floor_points, usable_floor The whole floor map and the array containing the roads
    # @param blocks A list of block types
    # @param package=None The package of lvl, start point and blocks for when connecting houses
    def connect_roads(self, rows, cols, level, floor_points, usable_floor, blocks, package=None):
        is_start, is_end = False, False
        block, slab, _ = blocks
        # Go through every path point
        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            x, y, z = floor_points[(r, c)]
            # Check if connecting houses to roads
            if package is not None:
                lvl, start, floor_blocks = package
                if [r, c] == start or i in [1, 2, 3]:  # If the first several blocks
                    if lvl == 0: y = y - 2
                    elif lvl == 1: y = y - 1
                    else: y = y
                if level.blockAt(x, y, z) in floor_blocks or level.blockAt(x, y-1, z) in floor_blocks: continue

            if level.blockAt(x, y, z) in [0, 8, 9]:  # Build Bridge
                uf.setBlock(level, block, x, y+1, z)
                if not is_start: self.build_bridge_walls(level, block, x, y, z)
                else: is_start = False
                is_end = True
            else:  # Build Road
                if is_end:
                    uf.setBlock(level, slab, x, y+1, z)
                    is_end = False
                uf.setBlock(level, block, x, y, z)
                try:  # Put a step in front of the bridge
                    if usable_floor[r+1][c+1] != len(usable_floor)-1 and usable_floor[r+1][c+1] == 1:
                        uf.setBlock(level, slab, x, y+1, z)
                        is_start = True
                except:
                    print("Out of bounds for slab - maybe the floor is invalid!")

    # This method builds the walls of any bridge
    # @param level The level provided by MCEdit
    # @param block The block type for the walls
    # @param x, y, z The coordinates
    def build_bridge_walls(self, level, block, x, y, z):
        wall_points = [(x+1, y+1, z+1), (x-1, y+1, z-1), (x+1, y+1, z-1), (x-1, y+1, z+1), (x, y+1, z+1), (x, y+1, z-1), (x+1, y+1, z), (x-1, y+1, z)]
        for p in wall_points:  # Build the block for each wall point
            if level.blockAt(p[0], p[1], p[2]) not in [block[0], 8, 9] and level.blockAt(p[0], p[1]-1, p[2]) in [0, 8, 9]:
                uf.setBlock(level, block, p[0], y+2, p[2])
        if level.blockAt(x, y+2, z) in [block[0]]: uf.setBlock(level, (0, 0), x, y+2, z)  # Remove wall if on path

    # This method gets if there are any disconnected roads
    # @param level The level provided by MCEdit
    # @param floor_points, usable_floor The whole floor map and the array containing the roads
    def get_components(self, level, usable_floor, floor_points):
        try:  # Check if the selected bottom has no air gaps
            ar_floor = np.array(usable_floor, dtype=np.uint8)
        except:
            print("Invalid floor has been selected!\nNo components will be found!")
            return None, None
        # Get any disconnected components from the road map array/image
        components, objects, _, _ = cv2.connectedComponentsWithStats(ar_floor, connectivity=8)

        if components < 3: return None, None  # The background + 2 separated roads = 3 components
        comp = np.arange(1, components)  # All components except the background
        components_map = {}
        for x in range(0, ar_floor.shape[0]):
            for y in range(0, ar_floor.shape[1]):
                posx, posy, posz = floor_points[(x, y)]
                if level.blockAt(posx, posy, posz) in [8, 9]: ar_floor[x][y] = 1  # put water in path array

                if objects[x][y] in comp:  # Record the positions of each component
                    components_map[objects[x][y]] = [x, y]
                    comp = np.delete(comp, np.argwhere(comp == objects[x][y]))
        return components_map, ar_floor

    # This method removes any detected lava
    # @param level The level provided by MCEdit
    # @param box The selected Minecraft region
    # @param x, y, z The coordinates of the block
    def remove_lava(self, level, x, y, z, box):
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

    # This method sets the bounding box of each tree on path
    # @param level The level provided by MCEdit
    # @param x, y, z The coordinates of the road
    def remove_tree(self, level, x, y, z):
        points_x, points_y, points_z = [], [y], []
        is_start = True
        posy = y+1
        y += 1
        while True:  # For each y (height) level, get the bounding box if a tree
            if level.blockAt(x, y, z) in [17, 81, 162] and is_start:  # Remove tree iff the tree base is on the road
                points_y.append(y)
                points_x = self.get_x_bound(level, x, y, z, points_x)
                points_z = self.get_z_bound(level, x, y, z, points_z)
                if level.blockAt(x, y+1, z) in [18, 161]: is_start = False
            elif level.blockAt(x, y, z) in [18, 161] and not is_start:  # Remove the leaves of the tree
                points_y.append(y)
                points_x = self.get_x_bound(level, x, y, z, points_x)
                points_z = self.get_z_bound(level, x, y, z, points_z)
            else: break
            y += 1
        self.remove_tree_from_bound(level, points_x, points_y, points_z, [x, posy, z])

    # This method goes through every point in the found
    # x, z, y boundary box and removes any trees found
    # @param level The level given from MCEdit
    # @param points_x, points_y, points_z The list of x, y, z points
    # @param road The coordinates of the current road
    def remove_tree_from_bound(self, level, points_x, points_y, points_z, road):
        if points_x and points_y and points_z:
            # Get the min and max of each direction of the box
            minx, maxx = min(points_x), max(points_x)
            miny, maxy = min(points_y), max(points_y)
            minz, maxz = min(points_z), max(points_z)

            first_tree, leaf = False, None
            last_tree, trees = [], {}
            # Remove Trees on Path using the found x, z, y boundary box
            for x in range(minx-2, maxx+2):
                for z in range(minz-2, maxz+2):
                    for y in range(miny-1, 350):
                        if level.blockAt(x, y, z) in [17, 18, 81, 161, 162]:
                            if level.blockAt(x, y, z) in [17, 81, 162] and not first_tree and [x, y, z] != road: trees[(x, z)] = y
                            if level.blockAt(x, y, z) in [17, 81, 162] and not first_tree and [x, y, z] == road:
                                first_tree = True
                                uf.setBlock(level, (0, 0), x, y, z)
                                last_tree.append((x, z))
                            elif level.blockAt(x, y, z) in [17, 81, 162] and first_tree:
                                if (x, z) in last_tree: uf.setBlock(level, (0, 0), x, y, z)
                                else: break
                            elif level.blockAt(x, y, z) in [18, 161]:
                                leaf = level.blockAt(x, y, z)
                                uf.setBlock(level, (0, 0), x, y, z)
                                first_tree = False

            # Grow back leaves on cut trees that are not on road
            for tree in trees.iterkeys():
                y = trees[tree]
                points = [(tree[0]+1, tree[1]+1), (tree[0]-1, tree[1]-1), (tree[0]+1, tree[1]-1),
                          (tree[0]-1, tree[1]+1), (tree[0], tree[1]+1), (tree[0]+1, tree[1]),
                          (tree[0]-1, tree[1]), (tree[0], tree[1]-1)]
                leaf = leaf if leaf is not None else 18  # Set default leaf
                for p in points: uf.setBlock(level, (leaf, 0), p[0], y, p[1])
                uf.setBlock(level, (leaf, 0), tree[0], y+1, tree[1])

    # Get the leaves of the tree for the X axis
    # @param level The level provided by MCEdit
    # @param x, y, z The coordinates to search from
    # @param points_x The list of found points on the X axis
    def get_x_bound(self, level, x, y, z, points_x):
        posx = x+1
        reverse = False
        while True:
            if reverse is False:  # Search one direction
                if level.blockAt(posx, y, z) in [18, 161]:
                    points_x.append(posx)
                    posx += 1
                else:
                    posx = x-1
                    reverse = True
            else:  # Search the other direction
                if level.blockAt(posx, y, z) in [18, 161]:
                    points_x.append(posx)
                    posx -= 1
                else: break
        return points_x

    # Get the leaves of the tree for the Z axis
    # @param level The level provided by MCEdit
    # @param x, y, z The coordinates to search from
    # @param points_z The list of found points on the Z axis
    def get_z_bound(self, level, x, y, z, points_z):
        posz = z+1
        reverse = False
        while True:
            if reverse is False:  # Search one direction
                if level.blockAt(x, y, posz) in [18, 161]:  # Removes all trees TODO: Move it so you remove only if on path way
                    points_z.append(posz)
                    posz += 1
                else:
                    posz = z-1
                    reverse = True
            else:  # Search the other direction
                if level.blockAt(x, y, posz) in [18, 161]:  # Removes all trees TODO: Move it so you remove only if on path way
                    points_z.append(posz)
                    posz -= 1
                else: break
        return points_z

"""
This is the main function that 
gets called after executing the filter in MCEdit
"""
def perform(level, box, options, block=None, need_return=False):
    road = Roads()
    road_network = road.process_road_network(box)  # Get the road as a Numpy array Image
    biom, road_blocks = get_biom(level, box)  # Get the biom and types of blocks
    floor, usable_floor = get_floor(level, box)  # Get the usable floor box region
    if block is not None: road_blocks[0] = block # Select Block types
    road.build_road(level, floor, road_network, box, usable_floor, road_blocks)  # Build and connect roads
    if need_return: return road

# This method connects a House (from the door position)
# To the closest road, by extending the road system.
# It should be called from another filter, and roads should
# already be present.
# @param level The level provided by MCEdit
# @param door_loc The location point of the door
# @param road The road class
# @param blocks A list of block types
def connect_houses_to_roads(level, door_loc, road, blocks):
    if road.floor is None or road.usable_floor is None or road.blocks is None:
        print("Road Floor is None!")
        return
    floor = dict((value, key) for key, value in road.floor.iteritems())  # Revert the point map
    usable_floor = np.array(road.usable_floor)
    if len(usable_floor.shape) != 2: return
    x, z, y = door_loc

    # Get the point in space of the door (Depending on the door position)
    if floor.get((x, y-2, z)) is not None:  # Beginning of door
        start = floor.get((x, y-2, z))
        lvl = abs((x, y, z)[1]) - abs((x, y-2, z)[1])  # 2
    elif floor.get((x, y-1, z)) is not None:  # Middle of door
        start = floor.get((x, y-1, z))
        lvl = abs((x, y, z)[1]) - abs((x, y-1, z)[1])  # 1
    elif floor.get((x, y, z)) is not None:  # Top of door
        start = floor.get((x, y, z))
        lvl = 0
    else: print("Path cannot be found!"); return
    start = [start[0], start[1]]

    # Get end point (First run => get closest)
    end, run = None, 1
    ending = [usable_floor.shape[0], usable_floor.shape[1]]
    beginning = [start[0], start[1]] if start[1] < ending[1] else [0, 0]
    while run < 3:
        for r in range(beginning[0], ending[0]):
            for c in range(beginning[1], ending[1]):
                if usable_floor[r][c] == 255: end = [r, c]
        if end is None: beginning = [0, 0]; run += 1
        else: break
    if end is None: return

    # Get the A* path
    path = star.search(np.zeros(usable_floor.shape), 1, start, end)
    if path is None: return

    rows, cols = np.where(np.array(path) != -1)  # Get the actual path positions
    package = [lvl, start, [block[0] for block in blocks]]
    road.connect_roads(rows, cols, level, road.floor, usable_floor, road.blocks, package)

# Get the floor points array and the floor map of coordinates
# @param level The level provided by MCEdit
# @param box The region selected in Minecraft
def get_floor(level, box):
    mapped_points = {}
    usable_floor = []
    pos_x, pos_y = 0, 0
    for x in range(box.minx, box.maxx):  # depth
        col = []
        for z in range(box.minz, box.maxz):  # width
            for y in range(box.maxy, box.miny-1, -1):  # height (col) but count the selected level
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

# This method  gets the biom and
# selects road block types given that information.
# It is based on probabilities
# @param level The level provided by MCEdit
# @param box The region selected in Minecraft
# @return road_biom, road_blocks The biom and road block types selected on majority voting
def get_biom(level, box):
    road_choices = []
    # Choice per slice
    for (chunk, slices, point) in level.getChunkSlices(box):
        bins = np.bincount(chunk.root_tag["Level"]["Biomes"].value)
        count = bins/float(len(chunk.root_tag["Level"]["Biomes"].value))
        bioms = np.flatnonzero(count)
        probs = {i:count[i] for i in bioms}
        road_choices.append(np.random.choice(probs.keys(), p=probs.values()))

    road_biom = max(road_choices, key=road_choices.count)
    road_blocks = get_road_blocks(road_biom)
    return road_biom, road_blocks

# All Minecraft blocks can be found here - https://minecraft-ids.grahamedgecombe.com/
# All Minecraft biomes can be found here - https://minecraft.fandom.com/wiki/Biome/ID
# From official Minecraft Wiki (Biomes and Blocks ID)
# This method is to be used if the filter is used alone
def get_road_blocks(road_biom):
    # Block, Slab, Stairs
    if road_biom in [2, 7, 16, 17, 27, 28, 36, 37, 38, 39, 130, 165, 166, 167]: return [(43, 0), (44, 0), (109, 0)]
    else: return [(1, 0), (44, 5), (67, 0)]
