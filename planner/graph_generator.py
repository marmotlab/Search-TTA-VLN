#######################################################################
# Name: graph_generator.py
#
# - Wrapper for graph.py
# - Sends the formatted inputs into graph.py to get useful info 
#######################################################################

import sys
if sys.modules['TRAINING']:
    from .parameter import *
else:
    from .test_parameter import *

import numpy as np
import shapely.geometry
from sklearn.neighbors import NearestNeighbors
from .node import Node
from .graph import Graph, a_star


class Graph_generator:
    def __init__(self, map_size, k_size, sensor_range, plot=False):
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points, self.grid_coords = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node = []
        self.nodes_list = []
        self.node_utility = None
        self.guidepost = None


    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []
        self.y = []


    def edge_clear(self, coords):
        node_index = str(self.find_index_from_coords(self.node_coords, coords))
        self.graph.clear_edge(node_index)


    def generate_graph(self, robot_belief, frontiers):
        self.edge_clear_all_nodes()
        free_area = self.free_area(robot_belief)

        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]

        self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)
        self.find_nearest_neighbor_all_nodes(self.node_coords, robot_belief)

        self.node_utility = []
        for coords in self.node_coords:
            node = Node(coords, frontiers, robot_belief)
            self.nodes_list.append(node)
            utility = node.utility
            self.node_utility.append(utility)
        self.node_utility = np.array(self.node_utility)

        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:,0] + self.node_coords[:,1]*1j
        for node in self.route_node:
            index = np.argwhere(x.reshape(-1) == node[0]+node[1]*1j)[0]
            self.guidepost[index] = 1

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost


    def update_graph(self, robot_belief, old_robot_belief, frontiers, old_frontiers):
        new_free_area = self.free_area((robot_belief - old_robot_belief > 0) * 255)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_node_coords = self.uniform_points[candidate_indices]
        self.node_coords = np.concatenate((self.node_coords, new_node_coords))

        old_node_to_update = []
        for coords in new_node_coords:
            neighbor_indices = self.find_k_neighbor(coords, self.node_coords, robot_belief)
            old_node_to_update += neighbor_indices
        old_node_to_update = set(old_node_to_update)
        for index in old_node_to_update:
            coords = self.node_coords[index]
            self.edge_clear(coords)
            self.find_k_neighbor(coords, self.node_coords, robot_belief)

        old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
        new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        observed_frontiers_index = np.where(
            np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
        new_frontiers_index = np.where(
            np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
        observed_frontiers = old_frontiers[observed_frontiers_index]
        new_frontiers = frontiers[new_frontiers_index]
        for node in self.nodes_list:
            if node.zero_utility_node is True:
                pass
            else:
                node.update_observable_frontiers(observed_frontiers, new_frontiers, robot_belief)

        for new_coords in new_node_coords:
            node = Node(new_coords, frontiers, robot_belief)
            self.nodes_list.append(node)

        self.node_utility = []
        for i, coords in enumerate(self.node_coords):
            utility = self.nodes_list[i].utility
            self.node_utility.append(utility)
        self.node_utility = np.array(self.node_utility)

        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:, 0] + self.node_coords[:, 1] * 1j
        for node in self.route_node:
            index = np.argwhere(x.reshape(-1) == node[0] + node[1] * 1j)
            self.guidepost[index] = 1

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost


    def generate_uniform_points(self):
        padding_x = 0.5 * (self.map_x / NUM_COORDS_WIDTH)
        padding_y = 0.5 * (self.map_y / NUM_COORDS_HEIGHT)
        x = np.linspace(padding_x, self.map_x - padding_x - 1, NUM_COORDS_WIDTH).round().astype(int)
        y = np.linspace(padding_y, self.map_y - padding_y - 1, NUM_COORDS_HEIGHT).round().astype(int)

        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        matrix = np.stack((t1, t2), axis=-1)
        return points, matrix


    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free


    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords


    def find_k_neighbor(self, coords, node_coords, robot_belief):
        dist_list = np.linalg.norm((coords-node_coords), axis=-1)
        sorted_index = np.argsort(dist_list)
        k = 0
        neighbor_index_list = []
        while k < self.k_size and k< node_coords.shape[0]:
            neighbor_index = sorted_index[k]
            neighbor_index_list.append(neighbor_index)
            dist = dist_list[k]
            start = coords
            end = node_coords[neighbor_index]
            if not self.check_collision(start, end, robot_belief):
                a = str(self.find_index_from_coords(node_coords, start))
                b = str(neighbor_index)
                self.graph.add_node(a)
                self.graph.add_edge(a, b, dist)

                if self.plot:
                    self.x.append([start[0], end[0]])
                    self.y.append([start[1], end[1]])
            k += 1
        return neighbor_index_list


    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][:]]):
                start = p
                end = neighbour
                if not self.check_collision(start, end, robot_belief):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j])

                    if self.plot:
                        self.x.append([p[0], neighbour[0]])
                        self.y.append([p[1], neighbour[1]])


    def find_nearest_neighbor_all_nodes(self, node_coords, robot_belief):
        for i, p in enumerate(node_coords):
            filtered_coords = self.get_neighbors_grid_coords(p)

            for j, neighbour in enumerate(filtered_coords):
                start = p
                end = neighbour
                if not self.check_collision(start, end, robot_belief):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, np.linalg.norm(start-end))

                    if self.plot:
                        self.x.append([p[0], neighbour[0]])
                        self.y.append([p[1], neighbour[1]])


    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]


    def find_closest_index_from_coords(self, node_coords, p):
        return np.argmin(np.linalg.norm(node_coords - p, axis=1))


    def find_index_from_grid_coords_2d(self, p):
        diffs = np.linalg.norm(self.grid_coords - p, axis=2)  
        indices = np.where(diffs < 1e-5)
        
        if indices[0].size > 0:
            return indices[0][0], indices[1][0]
        else:
            raise ValueError(f"Coordinate {p} not found in self.grid_coords.")


    def find_closest_index_from_grid_coords_2d(self, p):
        distances = np.linalg.norm(self.grid_coords - p, axis=2)  
        flat_index = np.argmin(distances)
        return np.unravel_index(flat_index, distances.shape)
    

    def check_collision(self, start, end, robot_belief):
        collision = False
        line = shapely.geometry.LineString([start, end])

        sortx = np.sort([start[0], end[0]])
        sorty = np.sort([start[1], end[1]])

        robot_belief = robot_belief[sorty[0]:sorty[1]+1, sortx[0]:sortx[1]+1]

        occupied_area_index = np.where(robot_belief == 1)
        occupied_area_coords = np.asarray([occupied_area_index[1]+sortx[0], occupied_area_index[0]+sorty[0]]).T
        unexplored_area_index = np.where(robot_belief == 127)
        unexplored_area_coords = np.asarray([unexplored_area_index[1]+sortx[0], unexplored_area_index[0]+sorty[0]]).T
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


    def find_shortest_path(self, current, destination, node_coords):
        start_node = str(self.find_index_from_coords(node_coords, current))
        end_node = str(self.find_index_from_coords(node_coords, destination))
        route, dist = a_star(int(start_node), int(end_node), self.node_coords, self.graph)
        if start_node != end_node:
            assert route != []
        route = list(map(str, route))
        return dist, route

    def get_neighbors_grid_coords(self, coord):
        # Return the 8 closest neighbors of a given coordinate
        
        nearest_coord = self.node_coords[self.find_closest_index_from_coords(self.node_coords, coord)]
        rows, cols = self.grid_coords.shape[:2]
        neighbors = []
        i, j  = self.find_index_from_grid_coords_2d(nearest_coord)

        # Create a range of indices for rows and columns
        row_range = np.clip([i - 1, i, i + 1], 0, rows - 1)
        col_range = np.clip([j - 1, j, j + 1], 0, cols - 1)

        # Iterate over the valid indices
        for ni in row_range:
            for nj in col_range:
                if (ni, nj) != (i, j):  # Skip the center point
                    neighbors.append(tuple(self.grid_coords[ni, nj]))

        return neighbors