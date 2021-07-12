import keras
import numpy as np
from PIL import Image
import cv2.cv2 as cv2

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
    else: return [1, 44.5, 67]

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