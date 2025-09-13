#######################################################################
# Name: node.py
#
# - Contains info per node on graph (edge)
# - Contains: Position, Utility, Visitation History
#######################################################################

import sys
if sys.modules['TRAINING']:
    from .parameter import *
else:
    from .test_parameter import *

import numpy as np
import shapely.geometry


class Node():
    def __init__(self, coords, frontiers, robot_belief):
        self.coords = coords
        self.observable_frontiers = []
        self.sensor_range = SENSOR_RANGE
        self.initialize_observable_frontiers(frontiers, robot_belief)
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def initialize_observable_frontiers(self, frontiers, robot_belief):
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < self.sensor_range - 10]
        for point in frontiers_in_range:
            collision = self.check_collision(self.coords, point, robot_belief)
            if not collision:
                self.observable_frontiers.append(point)

    def get_node_utility(self):
        return len(self.observable_frontiers)

    def update_observable_frontiers(self, observed_frontiers, new_frontiers, robot_belief):
        if observed_frontiers != []:
            observed_index = []
            for i, point in enumerate(self.observable_frontiers):
                if point[0] + point[1] * 1j in observed_frontiers[:, 0] + observed_frontiers[:, 1] * 1j:
                    observed_index.append(i)
            for index in reversed(observed_index):
                self.observable_frontiers.pop(index)
        #
        if new_frontiers != []:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.sensor_range - 15]
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_belief)
                if not collision:
                    self.observable_frontiers.append(point)

        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self):
        self.observable_frontiers = []
        self.utility = 0
        self.zero_utility_node = True

    def check_collision(self, start, end, robot_belief):
        collision = False
        line = shapely.geometry.LineString([start, end])

        sortx = np.sort([start[0], end[0]])
        sorty = np.sort([start[1], end[1]])

        robot_belief = robot_belief[sorty[0]:sorty[1] + 1, sortx[0]:sortx[1] + 1]

        occupied_area_index = np.where(robot_belief == 1)
        occupied_area_coords = np.asarray(
            [occupied_area_index[1] + sortx[0], occupied_area_index[0] + sorty[0]]).T
        unexplored_area_index = np.where(robot_belief == 127)
        unexplored_area_coords = np.asarray(
            [unexplored_area_index[1] + sortx[0], unexplored_area_index[0] + sorty[0]]).T
        unfree_area_coords = np.concatenate((occupied_area_coords, unexplored_area_coords))

        for i in range(unfree_area_coords.shape[0]):
            coords = ([(unfree_area_coords[i][0], unfree_area_coords[i][1]),
                       (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1]),
                       (unfree_area_coords[i][0], unfree_area_coords[i][1] + 1),
                       (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1] + 1)])
            obstacle = shapely.geometry.Polygon(coords)
            collision = line.intersects(obstacle)
            if collision:
                break

        return collision
