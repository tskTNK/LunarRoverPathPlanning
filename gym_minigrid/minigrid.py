import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *
# import matlab.engine
import random
import cv2
# from ai import cs
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
from .simulation_param import max_env_steps_for_graphmaking, xlength, ylength, timePerStep, landingLatitudeDeg, landingLongitudeDeg, lunarRotationCycle, P_s, absorptivitySurface, emissivitySurface, stephanBoltz, TopoX, TopoY, TopoZ 

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# TP calc parameters
TPcalc_ON = True
TP_graph = False
Time_variant_only = False
NewPosCode = False

# Reward param
thermalThresholdUpper = 44+273.15
thermalThresholdLower = 0+273.15
thermalControlThreshold = 40
thermalPowerFactor = 10
Kt = 2
batteryThresholdMin = 60
batteryThresholdMax = 100
batteryControlThreshold = 37
batteryPowerFactor = 10
Kp = 2
slope_max_threshold = 15
exceedingSlopePenalty = 20
Ks = 0.01
lunarSurfaceTempThresholdUpper = 85+273.15
lunarSurfaceTempThresholdLower = 0+273.15
exceedingPenalty_ls = 20
K_ls = 0.025
timepenalty_def = 0.1
Kpos = 5
goalreward_def = 100
reward_all_scaler = 1
stay_reward = 1 # to reduce the demerit of stay
reward_range = (-4000, 2000)


# initial and goal position
agent_start_pos_def=(5,5)
agent_goal_pos_def=(-5,-5)

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class Node:

    def __init__(self, temp=0, radiationIn=0, conductionIn=0, heatInput=0, \
                heatCapacitance=0, typeConstant=0, emissivity=0, absorptivity=0, \
                spheat=0, density=0, surface=0, volume=0, mass=0):
        self.temp = temp
        self.radiationIn = radiationIn
        self.conductionIn = conductionIn
        self.heatInput = heatInput
        self.heatCapacitance = heatCapacitance
        self.typeConstant = typeConstant
        self.emissivity = emissivity
        self.absorptivity = absorptivity
        self.spheat = spheat
        self.density = density
        self.surface = surface
        self.volume = volume
        self.mass = mass

class Connection:

    def __init__(self, emissivity1=1, emissivity2=1, radSurface=0, viewFactor=0, \
                thermalContactResistance=0, thermalContactArea=0, innerThermalConductivity=0, \
                innerConductiveArea=0, innerConductiveLength=1, temp1=0, temp2=0):
        self.emissivity1 = emissivity1
        self.emissivity2 = emissivity2
        self.radSurface = radSurface
        self.viewFactor = viewFactor
        self.thermalContactResistance = thermalContactResistance
        self.thermalContactArea = thermalContactArea
        self.innerThermalConductivity = innerThermalConductivity
        self.innerConductiveArea = innerConductiveArea
        self.innerConductiveLength = innerConductiveLength
        self.temp1 = temp1
        self.temp2 = temp2

        stephanBoltz = 5.67e-8
        # qrad = -1*stephanBoltz*radSurface*viewFactor*emissivity1*emissivity2*(np.power(temp1, 4)-np.power(temp2, 4))
        qrad = -1*stephanBoltz*radSurface*viewFactor*(1/(1/emissivity1+1/emissivity2-1))*(np.power(temp1, 4)-np.power(temp2, 4)) # surface facing in parallel

        self.Qrad1 = qrad
        self.Qrad2 = -1*qrad

        cond = -1*thermalContactResistance*thermalContactArea*(temp1-temp2)
        cond2 = -1*innerThermalConductivity*innerConductiveArea/innerConductiveLength*(temp1-temp2)

        self.Qcond1 = cond+cond2
        self.Qcond2 = -1*(cond+cond2)

def map(nodeId1, nodeId2, nodeLength):
    return nodeId1*nodeLength + nodeId2

class Panel:

    def __init__(self, absorptivity=0, emissivity=0, surfaceArea=0, \
                thickness=0, spheat=0, density=0, viewFactor2moon=0, \
                elecDissip=0, theta=0, phi=0):
        self.absorptivity = absorptivity
        self.emissivity = emissivity
        self.surfaceArea = surfaceArea
        self.thickness = thickness
        self.spheat = spheat
        self.density = density
        self.viewFactor2space = 1-viewFactor2moon
        self.viewFactor2moon = viewFactor2moon
        self.elecDissip = elecDissip
        self.theta = theta
        self.phi = phi

    def normalVec(self, roverOrientation, X, Y, Z):
        vn = np.multiply(X, np.cos(np.deg2rad(roverOrientation + self.phi))) + np.multiply(Y, np.sin(np.deg2rad(roverOrientation + self.phi)))
        vn = np.multiply(vn, np.cos(np.deg2rad(self.theta))) + np.multiply(Z, np.sin(np.deg2rad(self.theta)))
        return vn

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1  = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction

            rot = 0 # toshiki
            if agent_dir == 1:
                rot = agent_dir-1
            elif agent_dir == 3:
                rot = agent_dir-1
            elif agent_dir == 0:
                rot = agent_dir+1
            elif agent_dir == 2:
                rot = agent_dir+1
            else:
                pass

            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*rot)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img


    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        slope=None, # toshiki
        apos_history=None,
        step_count=None, # toshiki
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                # img[ymin:ymax, xmin:xmax, :] = tile_img
                img[xmin:xmax, ymin:ymax, :] = tile_img

                if cell == None: # toshiki
                    if  agent_here == 0:
                        # slope_max_value = np.abs(np.max(slope))
                        slope_max_value = slope_max_threshold
                        slope_abs_value = np.abs(slope[i-1][j-1])
                        clr = slope_abs_value/slope_max_value*255
                        img[xmin:xmax, ymin:ymax, :] = [clr, clr, clr]

                if cell == None:
                    if  agent_here == 0:
                        if apos_history[i][j]==1:
                            # img[ymin:ymax, xmin:xmax, :] = [100, 100, 0]
                            img[xmin:xmax, ymin:ymax, :] = [0, 255, 0]
                            # img[xmin:xmax, ymin:ymax, :] = [0, 30, 0]

        filename = 'savedimage.jpg'
        cv2.imwrite(filename, img)

        # filename = 'roverpath%02d.jpg' % step_count
        # cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) #  OpenCV assumes the image to be BGR or BGRA (BGR is the default OpenCV colour format). This means that blue and red planes get flipped.

        return img

    def render_TP(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        slope=None, # toshiki
        apos_history=None, # toshiki
        lunarSurfaceTemp=None, # toshiki
        sunAngles_to_the_moon=None, 
        step_count=None, # toshiki
        lowtemp=None, # toshiki
        hightemp=None, # toshiki
        avetemp=None, # toshiki
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        print("time*****", step_count)
        print("lunar surface temp", lunarSurfaceTemp[0][7][step_count])

        lunarsurfacetemp_max_value = 100 # 95
        lunarsurfacetemp_min_value = 70 # 75
        lunarsurfacetemp_value_range = lunarsurfacetemp_max_value-lunarsurfacetemp_min_value
        # print("max threshold", lunarsurfacetemp_max_value)
        # print("min threshold", lunarsurfacetemp_min_value)

        lowtemp[step_count] = 0
        hightemp[step_count] = 0
        avetemp[step_count] = 0

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                # img[ymin:ymax, xmin:xmax, :] = tile_img
                img[xmin:xmax, ymin:ymax, :] = tile_img

                if cell == None: # toshiki
                    if  agent_here == 0:
                        if lunarSurfaceTemp[i-1][j-1][step_count] < lunarsurfacetemp_min_value:
                            lunarsurfacetemp_value = lunarsurfacetemp_min_value
                        elif lunarSurfaceTemp[i-1][j-1][step_count] > lunarsurfacetemp_max_value:
                            lunarsurfacetemp_value = lunarsurfacetemp_max_value
                        else:
                            lunarsurfacetemp_value = lunarSurfaceTemp[i-1][j-1][step_count]

                        if lunarSurfaceTemp[i-1][j-1][step_count] < lowtemp[step_count]:
                            lowtemp[step_count] = lunarSurfaceTemp[i-1][j-1][step_count]

                        if lunarSurfaceTemp[i-1][j-1][step_count] > hightemp[step_count]:
                            hightemp[step_count] = lunarSurfaceTemp[i-1][j-1][step_count]

                        avetemp[step_count] = avetemp[step_count] + lunarSurfaceTemp[i-1][j-1][step_count]

                        a = (lunarsurfacetemp_value-lunarsurfacetemp_min_value)/lunarsurfacetemp_value_range*255
                        img[xmin:xmax, ymin:ymax, :] = [a, 60, 255-a]

                        # red = (lunarsurfacetemp_value-lunarsurfacetemp_min_value)/lunarsurfacetemp_value_range*255
                        # blue = np.uint8((255-red))
                        # img[xmin:xmax, ymin:ymax, :] = [red, blue, blue]
                    else:
                        print("lunar surface temp here", lunarSurfaceTemp[i-1][j-1][step_count])

                # highlight shadowed grids 
                if cell == None:
                    if  agent_here == 0:
                        if sunAngles_to_the_moon[i-1][j-1][step_count]==90:
                            img[ymin:ymax, xmin:xmax, :] = [100, 100, 100]

                if cell == None:
                    if  agent_here == 0:
                        if apos_history[i][j]==1:
                            # img[ymin:ymax, xmin:xmax, :] = [100, 100, 0]
                            img[xmin:xmax, ymin:ymax, :] = [0, 0, 0]
        
        avetemp[step_count]=avetemp[step_count]/self.height/self.width

        print("temp_min_value", lowtemp[step_count])
        print("temp_max_value", hightemp[step_count])
        print("temp_ave_value", avetemp[step_count])

        filename = 'roverpath%02d.jpg' % step_count
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) #  OpenCV assumes the image to be BGR or BGRA (BGR is the default OpenCV colour format). This means that blue and red planes get flipped.

        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                if j > 0:
                    mask[i+1, j-1] = True
                    mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j] = True
                if j > 0:
                    mask[i-1, j-1] = True
                    mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask

class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        stay = 0
        upward = 1
        right = 2
        downward = 3
        left = 4

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=450,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size
        
        self.width = width
        self.height = height
        self.agent_start_pos_def = agent_start_pos_def
        self.agent_goal_pos_def = agent_goal_pos_def
               
        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = reward_range

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent (note: the smallest position is 1 not 0)
        self.agent_pos = None
        self.agent_dir = None

        # self.slope = np.zeros((self.width, self.height))  # toshiki
        self.apos_history = np.zeros((self.width, self.height), dtype='uint8')

        # Initialize the RNG
        self.seed(seed=seed)

        # Read slope map from csv
        self.read_environment()

        # Set TP calc variables
        self.rover_thermal_power_model_setup()

        self.lowtemp = np.zeros((len(self.timeVector)))
        self.hightemp = np.zeros((len(self.timeVector)))
        self.avetemp = np.zeros((len(self.timeVector)))

        print('initialization done')

        # Initialize the state
        self.reset()

    def read_environment(self):

        # CSVData = open("data.csv")
        # self.R_disp = np.loadtxt(CSVData, delimiter=",")

        # read slope data from csv
        CSVData2 = open("slopeSurfaceNormal.csv")
        slopeSurfaceNormal = np.loadtxt(CSVData2, delimiter=",")
        self.slopeSurfaceNormal = slopeSurfaceNormal.reshape(xlength,ylength,3)

        # read slope data from csv
        CSVData3 = open("../lunarSurfaceTemp.csv")
        lunarSurfaceTemp = np.loadtxt(CSVData3, delimiter=",")
        self.lunarSurfaceTemp = lunarSurfaceTemp.reshape(xlength,ylength,max_env_steps_for_graphmaking)

        CSVData4 = open("slopeSurfaceNormalVecs.csv")
        slopeSurfaceNormalVecs = np.loadtxt(CSVData4, delimiter=",")
        self.slopeSurfaceNormalVecs = slopeSurfaceNormalVecs.reshape(xlength,ylength,3)

        # read slope data from csv
        CSVData5 = open("Vsun.csv")
        self.Vsun = np.loadtxt(CSVData5, delimiter=",")

        # read slope data from csv
        CSVData6 = open("slope2dAngle.csv")
        slope2dAngle = np.loadtxt(CSVData6, delimiter=",")
        self.slope2dAngle = slope2dAngle.reshape(xlength,ylength,2)

        # read slope data from csv
        CSVData7 = open("slope.csv")
        self.slope = np.loadtxt(CSVData7, delimiter=",")
        # print(self.slope)
        
        # read slope data from csv
        CSVData8 = open("sunAngles_to_the_moon.csv")
        sunAngles_to_the_moon = np.loadtxt(CSVData8, delimiter=",")
        self.sunAngles_to_the_moon = sunAngles_to_the_moon.reshape(xlength,ylength,max_env_steps_for_graphmaking)


        self.TopoX = TopoX
        self.TopoY = TopoY
        self.TopoZ = TopoZ

        return 1

    def rover_thermal_power_model_setup(self):

        # Rover type
        RoverType = 2 # 1: well isolated heat connection, 2: moderate heat connection
        self.nodelength = 8
        self.panellength = 6

        # Set Node IDs
        self.top = 0
        self.side1 = 1
        self.side2 = 2
        self.front = 3
        self.rear = 4
        self.bottom = 5
        self.outerspace = 6
        self.lunarsurface = 7

        # Thermal calculation param
        self.acceleratingFactor = 240
        self.epochcut = 1
        self.roverOrientationResolution = 30
        self.roverOrientationAll = np.arange(0, 180, self.roverOrientationResolution, dtype=int)
        self.missionDaysNo = 10;
        self.missionDuration = self.missionDaysNo*24*60;  # minutes basis
        self.interval = timePerStep; # [minutes] -> it takes 'self.interval' minutes to move one grid
        self.timeVector = np.arange(0, self.missionDuration, self.interval, dtype=int)
        self.timevec = np.arange(0, self.missionDuration/self.interval+1, dtype=int)

        # Power calculation param
        self.solarPanelEff = 0.28
        self.solarPanelRatio2surfaceAera = 0.50
        self.solarPowerConversionEff = 0.80
        self.powerConversionLoss = 0.80
        self.maxBatteryPower = 3400*3.7*2 # [mWh], refering to HAKUTO specification
        self.initialChargingPercentage = 1

        self.panels = [Panel() for i in range(self.panellength)]

        # Top panel
        topSurface = 0.50 * 0.25 # length[m], width[m]
        electronicsPower = 4.5
        self.panels[self.top] = Panel(0.08, 0.95, topSurface, 0.01, 1.04, 1.8, 0, electronicsPower, 90, 0)

        # Side 1 panel
        sideSurface1 = 0.5 * 0.2 # length[m], width[m]
        self.panels[self.side1] = Panel(0.75, 0.81, sideSurface1, 0.01, 1.04, 1.8, 0.35, 0, 20, 90)

        # Side 2 panel
        sideSurface2 = 0.5 * 0.2 # length[m], width[m]
        self.panels[self.side2] = Panel(0.75, 0.81, sideSurface2, 0.01, 1.04, 1.8, 0.35, 0, 20, 270)

        # Front panel
        frontSurface = 0.25 * 0.2 # length[m], width[m]
        self.panels[self.front] = Panel(0.08, 0.95, frontSurface, 0.01, 1.04, 1.8, 0.39, 0, 15, 0)

        # Rear panel
        rearSurface = 0.25 * 0.2 # length[m], width[m]
        self.panels[self.rear] = Panel(0.08, 0.95, rearSurface, 0.01, 1.04, 1.8, 0.39, 0, 15, 180)

        # Bottom panel
        bottomSurface = 0.6 * 0.30 # length[m], width[m]
        self.panels[self.bottom] = Panel(0.05, 0.05, bottomSurface, 0.01, 1.04, 1.8, 1, 0, -90, 0)

        # Inner View Factor
                        #    TOP  SIDE1  SIDE2  FRONT  REAR  BOTTOM
        innerViewFactor = [ [0, 0.1, 0.1, 0.1, 0.1, 0.5],  # TOP
                            [0.1, 0, 0.5, 0.1, 0.1, 0.1],  # SIDE1
                            [0.1, 0.5, 0, 0.1, 0.1, 0.1],  # SIDE2
                            [0.1, 0.1, 0.1, 0, 0.5, 0.1],  # FRONT
                            [0.1, 0.1, 0.1, 0.5, 0, 0.1],  # REAR
                            [0.5, 0.1, 0.1, 0.1, 0.1, 0]] # BOTTOM

        self.nodes = [Node() for i in range(self.nodelength)]
        self.connections = [Connection() for i in range(self.nodelength*self.nodelength)]

        # Thermal Contact Resistance [W/m2/K]
        # 10000: metal to metal connection, 1000: standard isolation, 100: high isolation
        if RoverType  == 1:
            thermalCR_SIDE2TOP = 1
            thermalCR_BOTTOM2SIDE = 1
            thermalCR_BOTTOM2LunarSurface = 1
        elif RoverType == 2:
            thermalCR_SIDE2TOP = 1800
            thermalCR_BOTTOM2SIDE = 1800
            thermalCR_BOTTOM2LunarSurface = 300

        # Thermal Contact Surface [m2]
        thermalCS_SIDE2TOP = 0.01*0.1
        thermalCS_FR2TOP = 0.01*0.05
        thermalCS_BOTTOM2SIDE = 0.01*0.1
        thermalCS_BOTTOM2LunarSurface = 0.05*0.05*4

        # Emissivity
        self.nodes[self.top].emissivity= self.panels[self.top].emissivity
        self.nodes[self.side1].emissivity = self.panels[self.side1].emissivity
        self.nodes[self.side2].emissivity = self.panels[self.side2].emissivity
        self.nodes[self.front].emissivity = self.panels[self.front].emissivity
        self.nodes[self.rear].emissivity = self.panels[self.rear].emissivity
        self.nodes[self.bottom].emissivity = self.panels[self.bottom].emissivity
        self.nodes[self.outerspace].emissivity = 1.00
        self.nodes[self.lunarsurface].emissivity = emissivitySurface

        # Absorptivity
        self.nodes[self.top].absorptivity= self.panels[self.top].absorptivity
        self.nodes[self.side1].absorptivity = self.panels[self.side1].absorptivity
        self.nodes[self.side2].absorptivity = self.panels[self.side2].absorptivity
        self.nodes[self.front].absorptivity = self.panels[self.front].absorptivity
        self.nodes[self.rear].absorptivity = self.panels[self.rear].absorptivity
        self.nodes[self.bottom].absorptivity = self.panels[self.bottom].absorptivity

        # Specific heat [Ws/gK]
        self.nodes[self.top].spheat = self.panels[self.top].spheat
        self.nodes[self.side1].spheat = self.panels[self.side1].spheat
        self.nodes[self.side2].spheat = self.panels[self.side2].spheat
        self.nodes[self.front].spheat = self.panels[self.front].spheat
        self.nodes[self.rear].spheat = self.panels[self.rear].spheat
        self.nodes[self.bottom].spheat = self.panels[self.bottom].spheat

        # Density [g/cm3]
        self.nodes[self.top].density = self.panels[self.top].density
        self.nodes[self.side1].density = self.panels[self.side1].density
        self.nodes[self.side2].density = self.panels[self.side2].density
        self.nodes[self.front].density = self.panels[self.front].density
        self.nodes[self.rear].density = self.panels[self.rear].density
        self.nodes[self.bottom].density = self.panels[self.bottom].density

        # Volume [cm3]
        self.nodes[self.top].volume = self.panels[self.top].surfaceArea * self.panels[self.top].thickness *100*100*100
        self.nodes[self.side1].volume = self.panels[self.side1].surfaceArea * self.panels[self.side1].thickness *100*100*100
        self.nodes[self.side2].volume = self.panels[self.side2].surfaceArea * self.panels[self.side2].thickness *100*100*100
        self.nodes[self.front].volume = self.panels[self.front].surfaceArea * self.panels[self.front].thickness *100*100*100
        self.nodes[self.rear].volume = self.panels[self.rear].surfaceArea * self.panels[self.rear].thickness *100*100*100
        self.nodes[self.bottom].volume = self.panels[self.bottom].surfaceArea * self.panels[self.bottom].thickness *100*100*100

        # surface [m2]
        self.nodes[self.top].surface = self.panels[self.top].surfaceArea
        self.nodes[self.side1].surface = self.panels[self.side1].surfaceArea
        self.nodes[self.side2].surface = self.panels[self.side2].surfaceArea
        self.nodes[self.front].surface = self.panels[self.front].surfaceArea
        self.nodes[self.rear].surface = self.panels[self.rear].surfaceArea
        self.nodes[self.bottom].surface = self.panels[self.bottom].surfaceArea

        # Mass of elements
        for i in range(self.nodelength):
            self.nodes[i].mass = self.nodes[i].density*self.nodes[i].volume # [g]
            # print("volume", self.nodes[i].volume, "mass", self.nodes[i].mass, "spheat", self.nodes[i].spheat)

        # total mass
        self.totalmass = 0
        for i in range(self.nodelength):
            self.totalmass = self.totalmass + self.nodes[i].mass

        # heat Capacitanceacitance
        for i in range(self.panellength):
            self.nodes[i].heatCapacitance = self.nodes[i].spheat*self.nodes[i].mass # [Ws/K]
            # print("capacitance", self.nodes[i].heatCapacitance)

        self.nodes[self.outerspace].heatCapacitance = 1000000
        self.nodes[self.lunarsurface].heatCapacitance = 1000000

        # define constant-temperature nodes
        self.nodes[self.outerspace].typeConstant = 1
        self.nodes[self.lunarsurface].typeConstant = 1

        # Set radiative heat property and conductive heat property
        self.connections[map(self.top,self.outerspace,self.nodelength)].radSurface = self.panels[self.top].surfaceArea
        self.connections[map(self.top,self.outerspace,self.nodelength)].viewFactor = self.panels[self.top].viewFactor2space
        self.connections[map(self.side1,self.outerspace,self.nodelength)].radSurface = self.panels[self.side1].surfaceArea
        self.connections[map(self.side1,self.outerspace,self.nodelength)].viewFactor = self.panels[self.side1].viewFactor2space
        self.connections[map(self.side2,self.outerspace,self.nodelength)].radSurface = self.panels[self.side2].surfaceArea
        self.connections[map(self.side2,self.outerspace,self.nodelength)].viewFactor = self.panels[self.side2].viewFactor2space
        self.connections[map(self.front,self.outerspace,self.nodelength)].radSurface = self.panels[self.front].surfaceArea
        self.connections[map(self.front,self.outerspace,self.nodelength)].viewFactor = self.panels[self.front].viewFactor2space
        self.connections[map(self.rear,self.outerspace,self.nodelength)].radSurface = self.panels[self.rear].surfaceArea
        self.connections[map(self.rear,self.outerspace,self.nodelength)].viewFactor = self.panels[self.rear].viewFactor2space
        self.connections[map(self.bottom,self.outerspace,self.nodelength)].radSurface = self.panels[self.bottom].surfaceArea
        self.connections[map(self.bottom,self.outerspace,self.nodelength)].viewFactor = self.panels[self.bottom].viewFactor2space

        self.connections[map(self.top,self.lunarsurface,self.nodelength)].radSurface = self.panels[self.top].surfaceArea
        self.connections[map(self.top,self.lunarsurface,self.nodelength)].viewFactor = self.panels[self.top].viewFactor2moon
        self.connections[map(self.side1,self.lunarsurface,self.nodelength)].radSurface = self.panels[self.side1].surfaceArea
        self.connections[map(self.side1,self.lunarsurface,self.nodelength)].viewFactor = self.panels[self.side1].viewFactor2moon
        self.connections[map(self.side2,self.lunarsurface,self.nodelength)].radSurface = self.panels[self.side2].surfaceArea
        self.connections[map(self.side2,self.lunarsurface,self.nodelength)].viewFactor = self.panels[self.side2].viewFactor2moon
        self.connections[map(self.front,self.lunarsurface,self.nodelength)].radSurface = self.panels[self.front].surfaceArea
        self.connections[map(self.front,self.lunarsurface,self.nodelength)].viewFactor = self.panels[self.front].viewFactor2moon
        self.connections[map(self.rear,self.lunarsurface,self.nodelength)].radSurface = self.panels[self.rear].surfaceArea
        self.connections[map(self.rear,self.lunarsurface,self.nodelength)].viewFactor = self.panels[self.rear].viewFactor2moon
        self.connections[map(self.bottom,self.lunarsurface,self.nodelength)].radSurface = self.panels[self.bottom].surfaceArea
        self.connections[map(self.bottom,self.lunarsurface,self.nodelength)].viewFactor = self.panels[self.bottom].viewFactor2moon

        for ID1 in range(self.panellength): # 0-5
            for ID2 in range(ID1, self.panellength):
                # print("ID1, ID2", ID1, ID2)
                self.connections[map(ID1,ID2,self.nodelength)].radSurface = self.panels[ID1].surfaceArea
                self.connections[map(ID1,ID2,self.nodelength)].viewFactor = innerViewFactor[ID1][ID2]

        self.connections[map(self.top,self.side1,self.nodelength)].thermalContactArea = thermalCS_SIDE2TOP
        self.connections[map(self.top,self.side1,self.nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
        self.connections[map(self.top,self.side2,self.nodelength)].thermalContactArea = thermalCS_SIDE2TOP
        self.connections[map(self.top,self.side2,self.nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
        self.connections[map(self.top,self.front,self.nodelength)].thermalContactArea = thermalCS_FR2TOP
        self.connections[map(self.top,self.front,self.nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
        self.connections[map(self.top,self.rear,self.nodelength)].thermalContactArea = thermalCS_FR2TOP
        self.connections[map(self.top,self.rear,self.nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
        self.connections[map(self.bottom,self.side1,self.nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
        self.connections[map(self.bottom,self.side1,self.nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE
        self.connections[map(self.bottom,self.side2,self.nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
        self.connections[map(self.bottom,self.side2,self.nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE
        self.connections[map(self.bottom,self.front,self.nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
        self.connections[map(self.bottom,self.front,self.nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE
        self.connections[map(self.bottom,self.rear,self.nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
        self.connections[map(self.bottom,self.rear,self.nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE

        self.connections[map(self.bottom,self.lunarsurface,self.nodelength)].thermalContactArea = thermalCS_BOTTOM2LunarSurface
        self.connections[map(self.bottom,self.lunarsurface,self.nodelength)].thermalContactResistance = thermalCR_BOTTOM2LunarSurface

        for ID in range(self.panellength):
            self.nodes[ID].heatCapacitance = self.nodes[ID].heatCapacitance/self.acceleratingFactor
            # print("heatCapacitance", self.nodes[ID].heatCapacitance)

        self.roverAllTemp = np.zeros((self.panellength, len(self.timeVector)+1, len(self.roverOrientationAll)))
        self.roverAllTempSelected = np.zeros((self.panellength, len(self.timeVector)+1))
        self.roverTopTempSelected = np.zeros((len(self.timeVector)+1))
        self.powerGen = np.zeros((len(self.timeVector)+1, len(self.roverOrientationAll)))
        self.BatteryPower = np.zeros((len(self.timeVector)+1))
        self.BatteryPowerPercentage = np.zeros((len(self.timeVector)+1))

        self.rewardPos = np.zeros((len(self.timeVector)+1))
        self.slopepenalty = np.zeros((len(self.timeVector)+1))
        self.TPpenalty = np.zeros((len(self.timeVector)+1))
        self.thermalPenalty_ = np.zeros((len(self.timeVector)+1))
        self.powerPenalty_ = np.zeros((len(self.timeVector)+1))
        self.LStemppenalty = np.zeros((len(self.timeVector)+1))
        self.LStempHistory = np.zeros((len(self.timeVector)+1))
        self.actionHistory = np.zeros((len(self.timeVector)+1))
        self.SlopeHistory = np.zeros((len(self.timeVector)+1))

        return 1

    def reset(self):

        print('reset')

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None
        self.prev_shaping = None
        self.initial_pos = None
        self.goal_pos = None
        self.apos_history = np.zeros((self.width, self.height), dtype='uint8')

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        self.BatteryPower[self.step_count] = self.maxBatteryPower*self.initialChargingPercentage
        self.BatteryPowerPercentage[self.step_count] = self.BatteryPower[self.step_count]/self.maxBatteryPower*100

        # for i in range(self.panellength):
        #     self.roverAllTempSelected[i][self.step_count] = 280
        # self.roverTopTempSelected[self.step_count] = 280

        self.thermal_power_calculation(self.step_count)
        self.best_rover_pose_calculation(self.step_count)
        
        self.LStempHistory[self.step_count] = self.lunarSurfaceTemp[self.agent_pos[0]-1][self.agent_pos[1]-1][self.step_count] + 273.15
        self.SlopeHistory[self.step_count] = self.slope[self.agent_pos[0]-1][self.agent_pos[1]-1]

        # print("step_count", self.step_count)
        # print("pos", self.agent_pos[0],self.agent_pos[1])
        # print("BatteryPowerPercentage", self.BatteryPowerPercentage[self.step_count])
        # print("roverTopTempSelected", self.roverTopTempSelected[self.step_count])
        # print("LStempHistory", self.LStempHistory[self.step_count])

        # Return first observation
        obs = self.gen_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        # _goalreward = (self.grid.height-2+self.agent_goal_pos_def[0])*(self.grid.width-2+self.agent_goal_pos_def[1])/goal_reward_scaler
        _goalreward = goalreward_def

        return _goalreward

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=False,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4) # the original code also uses (0, 4), why not (0, 3)?

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of downward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - self.agent_view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - self.agent_view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - self.agent_view_size + 1
            topY = self.agent_pos[1] - self.agent_view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - self.agent_view_size // 2
            topY = self.agent_pos[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, action):
        self.step_count += 1

        reward = 0
        goalreward = 0
        done = False

        # removed slip function (10/24/2022)

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move left
        if action == self.actions.left:
            self.agent_dir = 2 # left

        # Move right
        elif action == self.actions.right:
            self.agent_dir = 0 # right

        # Move downward
        elif action == self.actions.downward:
            self.agent_dir = 1 # down   

        # Move upward
        elif action == self.actions.upward:
            self.agent_dir = 3 # upward

        # Stay action
        elif action == self.actions.stay:
            pass

        else:
            assert False, "unknown action"

        goal_flag = False

        # Motion
        if action == self.actions.stay:
            pass
        else:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                goalreward = self._reward()
                goal_flag = True

        # Count increment
        if self.step_count > self.max_steps-1:
            done = True

        t = self.step_count # note t starts from 1, not 0

        if TPcalc_ON:
            self.thermal_power_calculation(t)
            self.best_rover_pose_calculation(t)

        self.TPpenalty[t] = self.thermalPenalty_[t]+self.powerPenalty_[t]
        
        # calculate positioning reward
        self.initial_pos = [self.agent_start_pos_def[0], self.agent_start_pos_def[1]] 
        self.goal_pos = [self.width-2+self.agent_goal_pos_def[0], self.height-2+self.agent_goal_pos_def[1]]
        v_distance = self.goal_pos[0] - self.agent_pos[0]
        h_distance = self.goal_pos[1] - self.agent_pos[1]
        shaping = - np.sqrt(v_distance*v_distance + h_distance*h_distance)

        if self.prev_shaping is not None:
            self.rewardPos[t] = shaping - self.prev_shaping
        else:
            self.rewardPos[t] = shaping + math.dist(self.initial_pos, self.goal_pos)
            # print('no_shaping')

        self.rewardPos[t] = self.rewardPos[t]*Kpos
        self.prev_shaping = shaping

        # revised pos reward (11/21/2022)
        if NewPosCode == True:
            self.rewardPos[t] = 0
            if goal_flag == True or self.step_count == self.max_steps:
                self.rewardPos[t] = shaping + math.dist(self.initial_pos, self.goal_pos)
                self.rewardPos[t] = self.rewardPos[t]*Kpos

        # calculate slope penalty
        xpos = self.agent_pos[0]
        ypos = self.agent_pos[1]

        if action == self.actions.stay:
            self.slopepenalty[t] = 0
            # print('stay!')
        else:
            slope_value = self.slope[xpos-1][ypos-1]
            self.slopepenalty[t] = Ks*slope_value*slope_value
            if slope_value > slope_max_threshold:
                self.slopepenalty[t] = self.slopepenalty[t] + exceedingSlopePenalty

        # calculate time penalty
        timepenalty = timepenalty_def

        # calculate time-variant penalty
        lunarSurfaceTempOfGrid = self.lunarSurfaceTemp[self.agent_pos[0]-1][self.agent_pos[1]-1][t] + 273.15

        if lunarSurfaceTempOfGrid > (lunarSurfaceTempThresholdUpper + lunarSurfaceTempThresholdLower)/2:
            Tc_ls = lunarSurfaceTempThresholdLower
        else:
            Tc_ls = lunarSurfaceTempThresholdUpper

        if lunarSurfaceTempOfGrid < lunarSurfaceTempThresholdUpper and lunarSurfaceTempOfGrid > lunarSurfaceTempThresholdLower:
            self.LStemppenalty[t] = K_ls*abs(Tc_ls-lunarSurfaceTempOfGrid)
        else:
            self.LStemppenalty[t] = K_ls*abs(Tc_ls-lunarSurfaceTempOfGrid) + exceedingPenalty_ls

        self.LStempHistory[t] = lunarSurfaceTempOfGrid
        
        self.SlopeHistory[t] = self.slope[self.agent_pos[0]-1][self.agent_pos[1]-1]


        if Time_variant_only == True:
            reward = reward + self.rewardPos[t]
            reward = reward - self.slopepenalty[t]
            reward = reward - timepenalty
            reward = reward - self.LStemppenalty[t]
            reward = reward + goalreward
        else:
            reward = reward + self.rewardPos[t]
            reward = reward - self.slopepenalty[t]
            reward = reward - timepenalty
            reward = reward - self.TPpenalty[t]
            reward = reward + goalreward

        if action == self.actions.stay:
            reward = reward + stay_reward

        # print("...")
        # print("t", t)
        # print("xpos, ypos", xpos, ypos)
        # print("self.rewardPos", self.rewardPos[t])
        # print("self.slopepenalty", - self.slopepenalty[t])
        # print("timepenalty", - timepenalty)
        # print("self.TPpenalty", -self.TPpenalty[t])
        # print("self.thermalPenalty_", -self.thermalPenalty_[t])
        # print("self.powerPenalty_", -self.powerPenalty_[t])
        # print("self.LStemppenalty", -self.LStemppenalty[t])
        # print("goalreward", goalreward)

        reward = reward/reward_all_scaler

        self.actionHistory[t] = action

        obs = self.gen_obs()

        if self.apos_history[xpos][ypos] == 0:
            self.apos_history[xpos][ypos] = 1

        return obs, reward, done, {}

    def thermal_power_calculation(self, t):

        # temporary variable for TP calculation
        iterationTime = int(12000/self.acceleratingFactor)
        iterationTime2 = int(self.interval*60/self.acceleratingFactor/self.epochcut)
        nodeTemp = np.zeros((self.nodelength, iterationTime))
        nodeQtotal = np.zeros((self.nodelength, iterationTime))
        nodeHeatInput = np.zeros((self.nodelength, iterationTime))
        nodeRadInput = np.zeros((self.nodelength, iterationTime))
        nodeCondInput = np.zeros((self.nodelength, iterationTime))
        panelNormalVecs = np.zeros((3, self.panellength))
        sunAnglesToPanels = np.zeros((self.panellength))
        Q_in_sun = np.zeros((self.panellength))
        Q_in_la = np.zeros((self.panellength))
        Q_in_e = np.zeros((self.panellength))
        solarPowerDischarge = np.zeros((self.panellength))
        Q_total = np.zeros((self.panellength))
        MonitorQrad = np.zeros((self.nodelength, self.nodelength))
        MonitorQcond = np.zeros((self.nodelength, self.nodelength))

        # lunar surface temperature calculation
        self.nodes[self.outerspace].temp = 3;
        self.nodes[self.lunarsurface].temp = self.lunarSurfaceTemp[self.agent_pos[0]-1,self.agent_pos[1]-1,self.step_count] + 273.15; # degC -> K

        # creat Body vector at each epoch with regard to the cartesian frame
        rotAngleAroundTopoY = self.slope2dAngle[self.agent_pos[0]-1,self.agent_pos[1]-1,0]
        rotAngleAroundTopoX = self.slope2dAngle[self.agent_pos[0]-1,self.agent_pos[1]-1,1]
        # print("inclination", rotAngleAroundTopoX, rotAngleAroundTopoY)
        # rotate around the current Y axis by localLongitude
        rotationMat1LS =  [[np.cos(np.deg2rad(rotAngleAroundTopoY)), 0, np.sin(np.deg2rad(rotAngleAroundTopoY))],
                            [0, 1, 0],
                            [-np.sin(np.deg2rad(rotAngleAroundTopoY)), 0, np.cos(np.deg2rad(rotAngleAroundTopoY))]]
        # rotate around the current X axis by localLattitude
        rotationMat2LS =  [[1, 0, 0],
                            [0, np.cos(np.deg2rad(rotAngleAroundTopoX)), -np.sin(np.deg2rad(rotAngleAroundTopoX))],
                            [0, np.sin(np.deg2rad(rotAngleAroundTopoX)), np.cos(np.deg2rad(rotAngleAroundTopoX))]]
        # rotate frame with regard to the current frame
        rotationLS = np.dot(rotationMat1LS, rotationMat2LS)

        # print(rotationLS)

        SurfaceX = np.dot(self.TopoX, rotationLS)
        SurfaceY = np.dot(self.TopoY, rotationLS)
        SurfaceZ = np.dot(self.TopoZ, rotationLS)


        # rover tempearture/power calculation
        for pose in range(len(self.roverOrientationAll)):

            # initialization
            for i in range(self.nodelength):
                self.nodes[i].radiationIn = 0
                self.nodes[i].conductionIn = 0
                self.nodes[i].heatInput = 0

            # Set temoporary rover orientation
            roverOrientationTemp = self.roverOrientationAll[pose]

            # Calculate Sun Angle relative to panels
            for i in range(self.panellength):
                # create top vectors based on the current rover orientation and body vector
                panelNormalVecs[:,i] = self.panels[i].normalVec(roverOrientation=roverOrientationTemp, X=SurfaceX, Y=SurfaceY, Z=SurfaceZ)
                # cap the sun angle value less than 90 using min fucntion
                sunAngles_temporary = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(panelNormalVecs[:,i], self.Vsun[t])), np.dot(panelNormalVecs[:,i], self.Vsun[t])))
                sunAnglesToPanels[i] = min(sunAngles_temporary, 90)

            # Calculate Solar radiation
            for i in range(self.panellength):
                Q_in_sun[i] = self.panels[i].absorptivity * np.cos(np.deg2rad(sunAnglesToPanels[i])) * self.panels[i].surfaceArea * P_s

            # Detect shadow flag
            shadowFlag = 0
            if all(item == 90 for item in sunAnglesToPanels):
                shadowFlag = 1

            # Calculate Lunar albedo
            for i in range(self.panellength):
                Q_in_la[i] = self.panels[i].absorptivity * self.panels[i].viewFactor2moon * self.panels[i].surfaceArea * (1-absorptivitySurface) * P_s

            # Calculate Lunar radiation
            Pe = emissivitySurface*stephanBoltz*np.power(self.nodes[self.lunarsurface].temp, 4);
            for i in range(self.panellength):
                Q_in_e[i] = self.panels[i].emissivity * self.panels[i].viewFactor2moon * self.panels[i].surfaceArea * Pe

            # Calculate Elec Power
            electronicsPowerSum = 0;
            if shadowFlag == 0:
                for i in range(self.panellength):
                    electronicsPowerSum = electronicsPowerSum + self.panels[i].elecDissip

            # Solar power discharging
            if Q_in_sun[self.side1]*self.solarPanelEff*self.solarPanelRatio2surfaceAera- electronicsPowerSum > Q_in_sun[self.side2]*self.solarPanelEff*self.solarPanelRatio2surfaceAera:
                solarPowerDischarge[self.side1] = -electronicsPowerSum
            elif Q_in_sun[self.side2]*self.solarPanelEff*self.solarPanelRatio2surfaceAera- electronicsPowerSum > Q_in_sun[self.side1]*self.solarPanelEff*self.solarPanelRatio2surfaceAera:
                solarPowerDischarge[self.side2] = -electronicsPowerSum
            else:
                solarPowerDischarge[self.side1] = -electronicsPowerSum/2
                solarPowerDischarge[self.side2] = -electronicsPowerSum/2

            # Set initial temperature for multi epoch calculation
            if t == 0:
            # if t == 0 or t == 1:
                for i in range(self.panellength):
                    Q_total[i] = Q_in_sun[i] + Q_in_la[i] + Q_in_e[i] + self.panels[i].elecDissip + solarPowerDischarge[i]
                    self.nodes[i].temp = np.power(Q_total[i]/(self.panels[i].emissivity * self.panels[i].surfaceArea * stephanBoltz), 0.25)
            else:
                # carry over temperatures from the last epoch
                for i in range(self.panellength):
                    self.nodes[i].temp = self.roverAllTempSelected[i,t-1]

            # Thermal calculation with inner conduction and radiation
            # this inner cycle updates every 1 second

            if t == 0:
            # if t == 0 or t == 1:
                pass
            else:
                iterationTime = iterationTime2 # use the transient calc from the second epoch

            for epoch in range(iterationTime):

                # Initialize heat flow
                for i in range(self.nodelength):
                    self.nodes[i].radiationIn = 0
                    self.nodes[i].conductionIn = 0
                    self.nodes[i].heatInput = 0

                # Calculate Solar radiation
                for i in range(self.panellength):
                    self.nodes[i].heatInput = self.nodes[i].heatInput + Q_in_sun[i]

                # Detect shadow flag
                shadowFlag = 0
                if all(item == 90 for item in sunAnglesToPanels):
                    shadowFlag = 1

                # Calculate Lunar albedo
                for i in range(self.panellength):
                    self.nodes[i].heatInput = self.nodes[i].heatInput + Q_in_la[i]

                # Calculate Lunar radiation -> will be calculated in the form of radiation heat
                # Pe = emissivitySurface*stephanBoltz*np.power(self.nodes[self.lunarsurface].temp, 4);
                # for i in range(panellength):
                #     self.nodes[ID].heatInput = self.nodes[ID].heatInput + Q_in_e[ID]

                # Calculate Elec Power
                electronicsPowerSum = 0;
                if shadowFlag == 0:
                    for i in range(self.panellength):
                        self.nodes[i].heatInput = self.nodes[i].heatInput + self.panels[i].elecDissip
                        electronicsPowerSum = electronicsPowerSum + self.panels[i].elecDissip

                # Solar power discharging
                if Q_in_sun[self.side1]*self.solarPanelEff*self.solarPanelRatio2surfaceAera*self.solarPowerConversionEff- electronicsPowerSum > Q_in_sun[self.side2]*self.solarPanelEff*self.solarPanelRatio2surfaceAera*self.solarPowerConversionEff:
                    self.nodes[self.side1].heatInput = self.nodes[self.side1].heatInput - electronicsPowerSum
                elif Q_in_sun[self.side2]*self.solarPanelEff*self.solarPanelRatio2surfaceAera*self.solarPowerConversionEff- electronicsPowerSum > Q_in_sun[self.side1]*self.solarPanelEff*self.solarPanelRatio2surfaceAera*self.solarPowerConversionEff:
                    self.nodes[self.side2].heatInput = self.nodes[self.side2].heatInput - electronicsPowerSum
                else:
                    self.nodes[self.side1].heatInput = self.nodes[self.side1].heatInput - electronicsPowerSum/2
                    self.nodes[self.side2].heatInput = self.nodes[self.side2].heatInput - electronicsPowerSum/2

                # Radiation and conduction between nodes
                for ID1 in range(self.nodelength):
                    for ID2 in range(self.nodelength):
                        radsurface = self.connections[map(ID1,ID2,self.nodelength)].radSurface
                        viewfactor = self.connections[map(ID1,ID2,self.nodelength)].viewFactor
                        contactarea = self.connections[map(ID1,ID2,self.nodelength)].thermalContactArea
                        contactresistatce = self.connections[map(ID1,ID2,self.nodelength)].thermalContactResistance

                        if ID1 < self.panellength and ID2 < self.panellength:
                            self.connections[map(ID1,ID2,self.nodelength)] = Connection(self.nodes[ID1].emissivity/10, self.nodes[ID2].emissivity/10, radsurface, viewfactor, contactresistatce, contactarea, 0, 0, 1, self.nodes[ID1].temp, self.nodes[ID2].temp)
                        else:
                            self.connections[map(ID1,ID2,self.nodelength)] = Connection(self.nodes[ID1].emissivity, self.nodes[ID2].emissivity, radsurface, viewfactor, contactresistatce, contactarea, 0, 0, 1, self.nodes[ID1].temp, self.nodes[ID2].temp)

                # Calculate heat transfer for each node
                for i in range(self.nodelength):
                    for j in range(self.nodelength):
                        MonitorQrad[i][j]= self.connections[map(i,j,self.nodelength)].Qrad1
                        MonitorQcond[i][j]= self.connections[map(i,j,self.nodelength)].Qcond1
                        self.nodes[i].radiationIn = self.nodes[i].radiationIn + self.connections[map(i,j,self.nodelength)].Qrad1
                        self.nodes[i].conductionIn = self.nodes[i].conductionIn + self.connections[map(i,j,self.nodelength)].Qcond1
                        self.nodes[j].radiationIn = self.nodes[j].radiationIn + self.connections[map(i,j,self.nodelength)].Qrad2
                        self.nodes[j].conductionIn = self.nodes[j].conductionIn + self.connections[map(i,j,self.nodelength)].Qcond2

                # update temperature for each node
                for i in range(self.nodelength):
                    nodeTemp[i][epoch] = self.nodes[i].temp
                    nodeHeatInput[i][epoch] = self.nodes[i].heatInput
                    nodeRadInput[i][epoch] = self.nodes[i].radiationIn
                    nodeCondInput[i][epoch] = self.nodes[i].conductionIn
                    nodeQtotal[i][epoch] = self.nodes[i].radiationIn + self.nodes[i].conductionIn + self.nodes[i].heatInput
                    if self.nodes[i].typeConstant == 0:
                        self.nodes[i].temp = self.nodes[i].temp + nodeQtotal[i][epoch]/self.nodes[i].heatCapacitance
                    else:
                        self.nodes[i].temp = self.nodes[i].temp

                # end of epoch

            # Save data for each pose
            for i in range(self.panellength):
                self.roverAllTemp[i][t][pose] = self.nodes[i].temp

            # Power Generation
            powerGenerationOfEpoch = 0
            for i in range(self.side1, self.side2+1):
                # print("ID=", i)
                powerGenerationOfEpoch = powerGenerationOfEpoch + Q_in_sun[i]*self.solarPanelEff*self.solarPanelRatio2surfaceAera*self.solarPowerConversionEff
            powerConsumptionOfEpoch = electronicsPowerSum/self.powerConversionLoss
            # self.powerGen[t][pose] = powerGenerationOfEpoch-powerConsumptionOfEpoch
            self.powerGen[t][pose] = powerGenerationOfEpoch-powerConsumptionOfEpoch

            # end of pose

        # print("t, pos", t, self.agent_pos[0], self.agent_pos[1])
        # print("lunar surface", round(self.lunarSurfaceTemp[self.agent_pos[0]-1][self.agent_pos[1]-1][t]))
        # print("top", self.roverAllTemp[top,t,:])
        # print("powerGen", self.powerGen[t,:])
        # print("sunAngleToTop", sunAnglesToPanels[self.top])
        # print("side1", self.roverAllTemp[side1,t,:])
        # print("side2", self.roverAllTemp[side2,t,:])
        # print("front", self.roverAllTemp[front,t,:])
        # print("rear", self.roverAllTemp[rear,t,:])
        # print("bottom", self.roverAllTemp[bottom,t,:])
        # print("pose1: top, side1, side2, front, rear, bottom, power", round(self.roverAllTemp[top,t,1]), round(self.roverAllTemp[side1,t,1]), round(self.roverAllTemp[side2,t,1]), round(self.roverAllTemp[front,t,1]), round(self.roverAllTemp[rear,t,1]), round(self.roverAllTemp[bottom,t,1]), self.powerGen[t,1])

        return self.roverAllTemp, self.powerGen

    def best_rover_pose_calculation(self, t):

        # Find rover best posture calculation based on thermal and power penalty
        TPpenalty_selected = 0

        powerGenList = np.zeros(len(self.roverOrientationAll))
        batteryPCList = np.zeros(len(self.roverOrientationAll))
        TPpenalty = np.zeros(len(self.roverOrientationAll))

        for pose in range(len(self.roverOrientationAll)):
            powerGenList[pose] = self.powerGen[t][pose]

            if t == 0:
                batteryPCList[pose] = self.BatteryPower[t] + powerGenList[pose]*1000*(self.interval/60)
            else:
                batteryPCList[pose] = self.BatteryPower[t-1] + powerGenList[pose]*1000*(self.interval/60)

            batteryPCList[pose] = batteryPCList[pose]/self.maxBatteryPower*100
            if  batteryPCList[pose]>100:
                batteryPCList[pose]=100

        roverTopThermalList = np.zeros(len(self.roverOrientationAll))
        roverTopThermalList = self.roverAllTemp[self.top,t,:]
        roverTopThermalList = roverTopThermalList.reshape(len(self.roverOrientationAll))

        # Calc penalty
        for pose in range(len(self.roverOrientationAll)):
            T_top = roverTopThermalList[pose]
            Bat_p = batteryPCList[pose]
            
            if T_top > (thermalThresholdLower + thermalThresholdUpper)/2:
                Tc = thermalThresholdLower
            else:
                Tc = thermalThresholdUpper

            Bat_pc = batteryThresholdMax

            thermalPenalty = Kt*(abs(Tc-T_top)/thermalControlThreshold)**thermalPowerFactor
            powerPenalty = Kp*(abs(Bat_pc-Bat_p)/batteryControlThreshold)**batteryPowerFactor

            TPpenalty[pose] = thermalPenalty+powerPenalty

        TPpenalty_selected = min(TPpenalty)
        selected_pose = np.where(TPpenalty == TPpenalty_selected)

        roverAllThermalList = np.zeros(self.panellength)
        roverAllThermalList = self.roverAllTemp[:,t,selected_pose]
        roverAllThermalList = roverAllThermalList.reshape(self.panellength)

        # rover thermal/power status update
        self.roverAllTempSelected[:,t] = roverAllThermalList
        self.roverTopTempSelected[t] = self.roverAllTempSelected[self.top,t]
        self.BatteryPower[t] = batteryPCList[selected_pose]*self.maxBatteryPower/100
        self.BatteryPowerPercentage[t] = batteryPCList[selected_pose]

        # Final reward recalculation
        T_top = self.roverTopTempSelected[t]
        Bat_p = self.BatteryPowerPercentage[t]
        
        if T_top > (thermalThresholdLower + thermalThresholdUpper)/2:
            Tc = thermalThresholdLower
        else:
            Tc = thermalThresholdUpper

        Bat_pc = batteryThresholdMax
        
        thermalPenalty = Kt*(abs(Tc-T_top)/thermalControlThreshold)**thermalPowerFactor
        powerPenalty = Kp*(abs(Bat_pc-Bat_p)/batteryControlThreshold)**batteryPowerFactor

        TPpenalty_selected = thermalPenalty+powerPenalty
        self.thermalPenalty_[t] = thermalPenalty
        self.powerPenalty_[t] = powerPenalty

        # print("BatteryPowerPercentage",self.BatteryPowerPercentage[t])
        # print("roverTopTempSelected",self.roverTopTempSelected[t])
        # print("TPpenalty",TPpenalty)
        # print("batteryPCList",batteryPCList)
        # print("roverTopThermalList",roverTopThermalList)
        # print("selected_pose",selected_pose)

        return self.thermalPenalty_, self.powerPenalty_


    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        
        if TP_graph == True:

            # Render the whole grid
            img = self.grid.render_TP(
                tile_size,
                self.agent_pos,
                self.agent_dir,
                self.slope,  # toshiki
                self.apos_history,   # toshiki
                self.lunarSurfaceTemp, # toshiki
                self.sunAngles_to_the_moon,
                self.step_count, # toshiki
                self.lowtemp,
                self.hightemp,
                self.avetemp,
                highlight_mask=highlight_mask if highlight else None
            )

        else:

            # Render the whole grid
            img = self.grid.render(
                tile_size,
                self.agent_pos,
                self.agent_dir,
                self.slope,  # toshiki
                self.apos_history,   # toshiki
                self.step_count, # toshiki
                highlight_mask=highlight_mask if highlight else None
            )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return
