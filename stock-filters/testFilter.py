from __future__ import division
import time # for timing
from math import sqrt, tan, sin, cos, pi, ceil, floor, acos, atan, asin, degrees, radians, log, atan2, acos, asin
from random import *
from numpy import *
from pymclevel import alphaMaterials, MCSchematic, MCLevel, BoundingBox
from mcplatform import *
import utilityFunctions as uf
import numpy as np
class Seed():
    def __init__(self,x,z,fake):
        self.id = randint(-100000,100000)
        self.seed_pos = [x,z]
        self.x = x
        self.z = z
        self.closest_points = []
        self.closest_points_2d = []
        self.bottom = []
        # self.col = [randint(0,255),randint(0,255),randint(0,255)]
        self.block = (159, randint(0,15))
        #self.y = y
        self.edge_points = []

        self.houses = []

        self.ratio = 0
        self.fake = fake
        self.mine = False
        self.center_dist = 0


def get_closest_points(map_points, seeds):
    for point in map_points:
        min_dist = 10000000
        min_seed_index = 0

        for s in seeds:
            dist = math.sqrt(((s.seed_pos[0]-point[0])**2)+((s.seed_pos[1]-point[1])**2))
            if dist < min_dist:
                min_dist = dist
                min_seed_index = seeds.index(s)

        seeds[min_seed_index].closest_points.append(point)
        seeds[min_seed_index].closest_points_2d.append((point[0], point[1]))

def get_edges(seeds):
    for seed in seeds:
        #print("cloest",seed.closest_points[0:10])
        #print("cloest 2d",seed.closest_points_2d[0:10])
        for point in seed.closest_points:
            if (point[0]+1,point[1]) not in seed.closest_points_2d:
                seed.edge_points.append(point)
            elif (point[0]-1,point[1]) not in seed.closest_points_2d:
                seed.edge_points.append(point)
            elif (point[0],point[1]+1) not in seed.closest_points_2d:
                seed.edge_points.append(point)
            elif (point[0],point[1]-1) not in seed.closest_points_2d:
                seed.edge_points.append(point)



def get_edges_2d(points):
    edge = []

    for point in points:
        if (point[0]+1,point[1]) not in points:
            edge.append(point)
        elif (point[0]-1,point[1]) not in points:
            edge.append(point)
        elif (point[0],point[1]+1) not in points:
            edge.append(point)
        elif (point[0],point[1]-1) not in points:
            edge.append(point)

    return edge


def get_edge_3d(points):

    points_dict = {}

    for point in points:
        points_dict[(point[0], point[1])] = point[2]

    edge = get_edges_2d(points_dict)
    edge_3d = []

    for point in edge:
        edge_3d.append((point[0], point[1], points_dict[point]))

    return edge_3d


def get_next_edge(current,edge,path,seed,path_2d):


    min_d = 100000
    min_angle_point = ()

    #[(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    #[(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]

    for (i,j) in [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]+[(0,2),(1,2),(2,2),(2,1),(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,2)]:
        if (current[0]+i, current[1]+j, current[2]) in edge:
            if (current[0]+i, current[1]+j, current[2]) not in path:
                if (current[0]+i, current[1]+j, current[2]+1) not in path and (current[0]+i, current[1]+j, current[2]+2) not in path:
                    if (current[0]+i, current[1]+j) not in path_2d[len(path_2d)-5:]:

                        return (current[0]+i, current[1]+j, current[2])




def get_next_layer_edge(current, next_layer, path):
    #    potential_next_points = []
    #
    #    for i in range(-2,3):
    #        for j in range(-2,3):
    #            if (current[0]+i, current[1]+j, current[2]-1) in next_layer:
    #                potential_next_points.append((current[0]+i, current[1]+j, current[2]-1))
    #
    #
    #    return min(potential_next_points, key = lambda x: (x[0]**2) + (x[1]**2))

    for (i,j) in [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]:
        if (current[0]+i, current[1]+j, current[2]-1) in next_layer:
            if (current[0]+i, current[1]+j, current[2]) not in path:
                if (current[0]+i, current[1]+j, current[2]+1) not in path:
                    return (current[0]+i, current[1]+j, current[2]-1)

    for (i,j) in [(0,2),(1,2),(2,2),(2,1),(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,2)]:
        if (current[0]+i, current[1]+j, current[2]-1) in next_layer:
            if (current[0]+i, current[1]+j, current[2]) not in path:
                if (current[0]+i, current[1]+j, current[2]+1) not in path:
                    return (current[0]+i, current[1]+j, current[2]-1)

    #    for i in range(-2,3):
    #        for j in range(-2,3):
    #            if (current[0]+i, current[1]+j, current[2]-1) in next_layer:
    #                return (current[0]+i, current[1]+j, current[2]-1)



    #    if (current[0], current[1], current[2]-1) in next_layer:
    #        return (current[0], current[1], current[2]-1)

    #    if (current[0], current[1]+1, current[2]-1) in next_layer:
    #        return (current[0], current[1]+1, current[2]-1)
    #
    #    elif (current[0]+1, current[1]+1, current[2]-1) in next_layer:
    #        return (current[0]+1, current[1]+1, current[2]-1)
    #
    #    elif (current[0]+1, current[1], current[2]-1) in next_layer:
    #        return (current[0]+1, current[1], current[2]-1)
    #
    #    elif (current[0]+1, current[1]-1, current[2]-1) in next_layer:
    #        return (current[0]+1, current[1]-1, current[2]-1)
    #
    #    elif (current[0], current[1]-1, current[2]-1) in next_layer:
    #        return (current[0], current[1]-1, current[2]-1)
    #
    #    elif (current[0]-1, current[1]-1, current[2]-1) in next_layer:
    #        return (current[0]-1, current[1]-1, current[2]-1)
    #
    #    elif (current[0]-1, current[1], current[2]-1) in next_layer:
    #        return  (current[0]-1, current[1], current[2]-1)
    #
    #    elif (current[0]-1, current[1]+1, current[2]-1) in next_layer:
    #        return (current[0]-1, current[1]+1, current[2]-1)
    #

    print("finding next layer")


#    if (current[0], current[1]+1, current[2]-1) in next_layer:
#        return (current[0], current[1]+1, current[2]-1)
#
#    elif (current[0]-1, current[1]+1, current[2]-1) in next_layer:
#        return (current[0]+1, current[1]+1, current[2]-1)
#
#    elif (current[0]-1, current[1], current[2]-1) in next_layer:
#        return (current[0]+1, current[1], current[2]-1)
#
#    elif (current[0]-1, current[1]-1, current[2]-1) in next_layer:
#        return (current[0]+1, current[1]-1, current[2]-1)
#
#    elif (current[0], current[1]-1, current[2]-1) in next_layer:
#        return (current[0], current[1]-1, current[2]-1)
#
#    elif (current[0]+1, current[1]-1, current[2]-1) in next_layer:
#        return (current[0]-1, current[1]-1, current[2]-1)
#
#    elif (current[0]+1, current[1], current[2]-1) in next_layer:
#        return  (current[0]-1, current[1], current[2]-1)
#
#    elif (current[0]+1, current[1]+1, current[2]-1) in next_layer:
#        return (current[0]-1, current[1]+1, current[2]-1)






def get_bottom(level, box, min_y):
    #bottom_matrix = empty([box.maxx-box.minx, box.maxz-box.minz])
    bottom = {}
    print("Here")
    print(box.minx, box.maxx,box.minz, box.maxz, box.miny, box.maxy)
    for x in range(box.minx, box.maxx):
        for z in range(box.minz, box.maxz):
            for y in range(350, box.miny, -1):
                #print(level.blockAt(x,y,z))
                # print("Y" + str(y))
                if level.blockAt(x,y,z) in [11,10]:
                    uf.setBlock(level,(49,0), x, y, z)############################

                if level.blockAt(x,y,z) in [1,2,3,13,9,11,10,8]:
                    bottom[(x,z)] = y
                    break
                #     #print((x,z))
                #     #bottom.append([x,y,z])





    #print("gotten_botem",bottom)
    return bottom
def get_district_seeds(box):
    district_seeds = []

    center_x = int((box.maxx+box.minx)/2)
    center_z = int((box.maxz+box.minz)/2)

    max_r = min([box.maxx-center_x,box.maxz-center_z])
    #min_r = max([(box.maxx-center_x)*0.1,(box.maxz-center_z)*0.1])

    median_r = max([int((box.maxx-box.minx)/2), int((box.maxz-box.minz)/2)])
    #r_adder = int(median_r*0.2)

    for d in range(0,360,70): #40
        r = randint(0, int(max_r*0.5))
        x = center_x + int(r*sin(radians(d)))
        z = center_z + int(r*cos(radians(d)))
        #print("pos",x,z)
        new_seed = Seed(x,z,False)
        new_seed.center_dist = 1
        district_seeds.append(new_seed)

    for d in range(20,380,60): #30
        r = (randint(int(max_r*0.7), int(max_r*0.9)))
        x = center_x + int(r*sin(radians(d)))
        z = center_z + int(r*cos(radians(d)))
        #print("pos",x,z)
        new_seed = Seed(x,z,False)
        new_seed.center_dist = 2
        district_seeds.append(new_seed)


    #print("district_seeds",len(district_seeds))

    for x in range(box.minx,box.maxx,40):
        district_seeds.append(Seed(x,box.minz,True))
        district_seeds.append(Seed(x,box.maxz,True))

    for z in range(box.minz,box.maxz,40):
        district_seeds.append(Seed(box.minx,z,True))
        district_seeds.append(Seed(box.maxx,z,True))
    #
    return district_seeds
def get_usable_site(level, seeds, bottom):
    #print("getting usable site")
    # THIS DOESNT WORK IDK WHY
    for seed in seeds:
        seed.usable_site = seed.closest_points[:]

        for point in seed.usable_site:

            #print(level.blockAt(point[0], point[1], bottom[(point[0], point[1])]))



            if level.blockAt(point[0], point[1], bottom[(point[0], point[1])]) in [8,9,10,11]:
                seed.usable_site.remove(point)

            if level.blockAt(point[0], point[1], bottom[(point[0], point[1])]) in [10,11]:
                uf.setBlock(level,(49,0), point[0], point[2], point[1])
                #print("found lava")

def perform(level, box, options):
    print "This is the level: " + str(level)
    print "This is the box: " + str(box)
    bottom = get_bottom(level, box, 1)
    print "This is the bottom: " + str(bottom)
    log_types = []
    stair_types = []
    slab_types = []
    plank_types = []

    for (chunk, slices, point) in level.getChunkSlices(box):
        #print(point, chunk.root_tag["Level"]["Biomes"].value)
        count = np.bincount(chunk.root_tag["Level"]["Biomes"].value)
        print(chunk.root_tag["Level"]["Biomes"].value)
        print("Array of percentages: "+str(count/len(chunk.root_tag["Level"]["Biomes"].value)))
        count = count/len(chunk.root_tag["Level"]["Biomes"].value)
        count = np.flatnonzero(count)
        max = np.max(count)
        print(max)
        def f(x):
            if x > 0.7: return x
            else: return -1

        print("Occurances: "+str(np.where(count == f(max), count, np.random.choice(count))[0]))
        for val in chunk.root_tag["Level"]["Biomes"].value:
            # print("Biom: "+str(val))


            if val in [1,3,4,6,8,132]: #oak
                log_types.append("oak")
                #stair_types.append(53)
                #slab_types.append((126,0))
                #plank_types.append((5,0))
            if val in [21,22,23,149,151]: #jungle
                log_types.append("jungle")
                #stair_types.append(136)
                #slab_types.append(126,3)
                #plank_types.append((5,3))
            if val in [5,12,13,19,30,31,32,33,158,160,161,133]: #spuce
                log_types.append("spruce")
                #stair_types.append(134)
                #slab_types.append((126,1))
                #plank_types.append((5,1))
            if val in [6,29,157,134]: #dark
                log_types.append("dark")
                #stair_types.append(164)
                #slab_types.append((126,5))
                #plank_types.append((5,5))
            if val in [2,17,27,28,155,156]: #birch
                log_types.append("birch")
                #stair_types.append(135)
                #slab_types.append((126,2))
                #plank_types.append((5,2))
            if val in [35,36,37,38,39,163,164,165,166,167]: #acacia
                log_types.append("acacia")
                #stair_types.append(163)
                #slab_types.append((126,4))
                #plank_types.append((5,4))

    if log_types == []:
        log_types = ["oak"]



        #arr = chunk.root_tag["Level"]["Biomes"].value

    #print(bottom)

    #print("bottom", len(bottom))

    print("finding map bottom")

    map_points_3d = []
    map_points_2d = []
    for point in bottom:
        point_3d = (point[0],point[1],bottom[point])
        map_points_3d.append(point_3d)
        map_points_2d.append((point[0],point[1]))

    #print("map_points", len(map_points_3d))

    print("seeding bottom")
    district_seeds = get_district_seeds(box)

    #print(district_seeds)

    get_closest_points(map_points_3d, district_seeds)
    get_edges(district_seeds)

    for seed in district_seeds:
        if seed.fake:
            district_seeds[district_seeds.index(seed)] = None

    for i in range(district_seeds.count(None)):
        district_seeds.remove(None)



    get_usable_site(level, district_seeds, bottom)

    lamp_post_points = []


    print("drawing major paths")

    for seed in district_seeds:
        for point in seed.edge_points:
            # print "Point: "+str(point)
            # print "Seed: "+str(seed)
            print(point)
            # for x in range(-1,2): # Adds on the sides of the chosen area
            for z in range(-2, 2): # Fills up the whole selected area on the y axis
                uf.setBlockToGround(level, (0, 0), point[0], point[1], point[2] + 2, point[2])
                uf.setBlock(level, (98, 0), point[0], point[2], point[1]+z)
                if level.blockAt(point[0], point[2]+1, point[1]+z) != 0:
                    uf.setBlock(level, (0, 0), point[0], point[2] + 1, point[1]+z)
            #         #uf.setBlockToGround(level, (0,0), point[0], point[1], point[2]+2, point[2])
            #         for k in range(2):
            #             uf.setBlock(level,(0,0), point[0]+x, point[2]+k+1, point[1]+z)
            #         #if randint(0,2*(abs(x)+abs(z))) == 0:
            #         block = choice([(98,0),(98,0),(98,0),(98,0),(1,5),(4,0),(13,0),(98,1),(98,2)])
            #
            #         if level.blockAt(point[0]+x, point[2], point[1]+z) == 9:
            #             uf.setBlockIfEmpty(level,(44,5), point[0]+x, point[2]+1, point[1]+z)
            #         else:
            #             uf.setBlock(level,block, point[0]+x, point[2], point[1]+z)
