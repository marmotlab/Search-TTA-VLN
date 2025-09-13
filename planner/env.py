#######################################################################
# Name: env.py
#
# - Reads and processes training and test maps 
# - Processes rewards, new frontiers given action
# - Updates a graph representation of environment for input into network
#######################################################################

import sys
if sys.modules['TRAINING']:
    from .parameter import *
else:
    from .test_parameter import *

import os
import cv2
import copy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from scipy.ndimage import label, find_objects
from .sensor import *
from .graph_generator import *
from .node import *


class Env():
    def __init__(self, map_index, n_agent, k_size=20, plot=False, test=False, mask_index=None):
        self.n_agent = n_agent
        self.test = test
        self.map_dir = GRIDMAP_SET_DIR 

        # Import environment gridmap
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)

        # NEW: Import segmentation utility map
        self.seg_dir = MASK_SET_DIR    
        self.segmentation_mask, self.target_positions, self.target_found_idxs = None, [], []
        self.segmentation_mask_list = os.listdir(self.seg_dir)
        self.segmentation_mask_list.sort(reverse=True)

        # # NEW: Find common files in both directories
        self.map_index = map_index % len(self.map_list)
        if mask_index is not None:
            self.mask_index = mask_index % len(self.segmentation_mask_list)
        else:
            self.mask_index = map_index % len(self.segmentation_mask_list)

        # Import ground truth and segmentation mask
        self.ground_truth, self.map_start_position = self.import_ground_truth(
            os.path.join(self.map_dir, self.map_list[self.map_index]))
        self.ground_truth_size = np.shape(self.ground_truth)  
        self.robot_belief = np.ones(self.ground_truth_size) * 127  # unexplored 127
        self.downsampled_belief = None
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.coverage_belief = np.ones(self.ground_truth_size) * 127  # unexplored 127

        # Import segmentation mask
        mask_filename = self.segmentation_mask_list[self.mask_index]
        self.segmentation_mask = self.import_segmentation_mask(
            os.path.join(self.seg_dir, mask_filename))
        
        # Overwrite target positions if directory specified
        if self.test and TARGETS_SET_DIR != "":
            self.target_positions = self.import_targets(
                os.path.join(TARGETS_SET_DIR, self.map_list[self.map_index])) 
        
        self.segmentation_info_mask = None
        self.segmentation_info_mask_unnormalized = None
        self.filtered_seg_info_mask = None
        self.num_targets_found = 0
        self.num_new_targets_found = 0
        self.resolution = 4
        self.sensor_range = SENSOR_RANGE
        self.explored_rate = 0
        self.targets_found_rate = 0
        self.frontiers = None
        self.start_positions = []
        self.plot = plot
        self.frame_files = []
        self.graph_generator = Graph_generator(map_size=self.ground_truth_size, sensor_range=self.sensor_range, k_size=k_size, plot=plot)
        self.node_coords, self.graph, self.node_utility, self.guidepost = None, None, None, None

        self.begin(self.map_start_position)


    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index

    def begin(self, start_position):
        self.robot_belief = self.ground_truth   
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        self.frontiers = self.find_frontier()
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        self.node_coords, self.graph, self.node_utility, self.guidepost = self.graph_generator.generate_graph(
                self.robot_belief, self.frontiers)
        
        # Define start positions
        if FIX_START_POSITION:
            coords_res_row = int(self.robot_belief.shape[0]/NUM_COORDS_HEIGHT)
            coords_res_col = int(self.robot_belief.shape[1]/NUM_COORDS_WIDTH)
            self.start_positions = [(int(self.robot_belief.shape[1]/2)-coords_res_col/2,int(self.robot_belief.shape[0]/2)-coords_res_row/2)  for _ in range(self.n_agent)]   
        else:
            nearby_coords = self.graph_generator.get_neighbors_grid_coords(start_position)
            itr = 0
            for i in range(self.n_agent):
                if i == 0 or len(nearby_coords) == 0:
                    self.start_positions.append(start_position)
                else:
                    idx = min(itr, len(nearby_coords)-1)
                    self.start_positions.append(nearby_coords[idx])
                    itr += 1

        for i in range(len(self.start_positions)):
            self.start_positions[i] = self.node_coords[self.find_index_from_coords(self.start_positions[i])]
            self.coverage_belief = self.update_robot_belief(self.start_positions[i], self.sensor_range, self.coverage_belief,
                                                        self.ground_truth)

        for start_position in self.start_positions:
            self.graph_generator.route_node.append(start_position)

        # Info map from ground truth
        rng_x = 0.5 * (self.ground_truth.shape[1] / NUM_COORDS_WIDTH)
        rng_y = 0.5 * (self.ground_truth.shape[0] / NUM_COORDS_HEIGHT)
        self.segmentation_info_mask = np.zeros((len(self.node_coords), 1))
        for i, node_coord in enumerate(self.node_coords):
            max_x = min(node_coord[0] + int(math.ceil(rng_x)), self.ground_truth.shape[1])
            min_x = max(node_coord[0] - int(math.ceil(rng_x)), 0)
            max_y = min(node_coord[1] + int(math.ceil(rng_y)), self.ground_truth.shape[0])
            min_y = max(node_coord[1] - int(math.ceil(rng_y)), 0)

            if TARGETS_SET_DIR == "":   
                exclude = {208} # Exclude target positions 
            else:
                exclude = {}
            self.segmentation_info_mask[i] = max(x for x in self.segmentation_mask[min_y:max_y, min_x:max_x].flatten() if x not in exclude) / 100.0

        self.filtered_seg_info_mask = copy.deepcopy(self.segmentation_info_mask)
        done, num_targets_found = self.check_done()
        self.num_targets_found = num_targets_found


    def multi_robot_step(self, next_position_list, dist_list, travel_dist_list):
        reward_list = []
        for dist, robot_position in zip(dist_list, next_position_list):
            self.graph_generator.route_node.append(robot_position)
            next_node_index = self.find_index_from_coords(robot_position)
            self.graph_generator.nodes_list[next_node_index].set_visited()
            self.coverage_belief = self.update_robot_belief(robot_position, self.sensor_range, self.coverage_belief,
                                                         self.ground_truth)
            self.robot_belief = self.ground_truth   
            self.downsampled_belief = block_reduce(self.robot_belief.copy(),
                                                   block_size=(self.resolution, self.resolution),
                                                   func=np.min)

            frontiers = self.find_frontier()
            individual_reward = -dist / 32 

            info_gain_reward = 0
            robot_position_idx = self.find_index_from_coords(robot_position)
            info_gain_reward = self.filtered_seg_info_mask[robot_position_idx][0]  * 1.5
            if self.guidepost[robot_position_idx] == 0.0:
                info_gain_reward += 0.2
            individual_reward += info_gain_reward

            reward_list.append(individual_reward)

        self.node_coords, self.graph, self.node_utility, self.guidepost = self.graph_generator.update_graph(self.robot_belief, self.old_robot_belief, frontiers, self.frontiers)
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        self.filtered_seg_info_mask = [info[0] if self.guidepost[i] == 0.0 else 0.0 for i, info in enumerate(self.segmentation_info_mask)]
        self.filtered_seg_info_mask = np.expand_dims(np.array(self.filtered_seg_info_mask), axis=1)

        self.frontiers = frontiers
        self.explored_rate = self.evaluate_exploration_rate()

        done, num_targets_found = self.check_done()
        self.num_new_targets_found = num_targets_found - self.num_targets_found
        team_reward = 0.0

        self.num_targets_found = num_targets_found
        self.targets_found_rate = self.evaluate_targets_found_rate()

        if done:
            team_reward += 40 
        for i in range(len(reward_list)):
            reward_list[i] += team_reward

        return reward_list, done


    def import_ground_truth(self, map_index):
        # occupied 1, free 255, unexplored 127

        try:
            ground_truth = (io.imread(map_index, 1)).astype(int)
            if np.all(ground_truth == 0):
                ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        except:
            new_map_index = self.map_dir + '/' + self.map_list[0]
            ground_truth = (io.imread(new_map_index, 1)).astype(int)
            print('could not read the map_path ({}), hence skipping it and using ({}).'.format(map_index, new_map_index))

        robot_location = np.nonzero(ground_truth == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location


    def import_segmentation_mask(self, map_index):
        mask = cv2.imread(map_index).astype(int)
        return mask 

    def import_targets(self, map_index):
        # occupied 1, free 255, unexplored 127, target 208
        mask = cv2.imread(map_index).astype(int)
        target_positions = self.find_target_locations(mask)
        return target_positions


    def find_target_locations(self, image_array, grey_value=208):

        grey_pixels = np.where(image_array == grey_value)
        binary_array = np.zeros_like(image_array, dtype=bool)
        binary_array[grey_pixels] = True
        labeled_array, num_features = label(binary_array)
        slices = find_objects(labeled_array)

        # Calculate the center of each box
        centers = []
        for slice in slices:
            row_center = (slice[0].start + slice[0].stop - 1) // 2
            col_center = (slice[1].start + slice[1].stop - 1) // 2
            centers.append((col_center, row_center))    # (y,x)

        return centers

    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief


    def check_done(self):
        done = False
        num_targets_found = 0
        self.target_found_idxs = []
        for i, target in enumerate(self.target_positions):
            if self.coverage_belief[target[1], target[0]] == 255: 
                num_targets_found += 1
                self.target_found_idxs.append(i)

        if TERMINATE_ON_TGTS_FOUND and num_targets_found >= len(self.target_positions):
            done = True
        if not TERMINATE_ON_TGTS_FOUND and np.sum(self.coverage_belief == 255) / np.sum(self.ground_truth == 255) >= 0.99:
            done = True
        
        return done, num_targets_found


    def calculate_num_observed_frontiers(self, old_frontiers, frontiers):
        frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        pre_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
        frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
        pre_frontiers_num = pre_frontiers_to_check.shape[0]
        delta_num = pre_frontiers_num - frontiers_num

        return delta_num

    def calculate_reward(self, dist, frontiers):
        reward = 0
        reward -= dist / 64

        frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        pre_frontiers_to_check = self.frontiers[:, 0] + self.frontiers[:, 1] * 1j
        frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
        pre_frontiers_num = pre_frontiers_to_check.shape[0]
        delta_num = pre_frontiers_num - frontiers_num

        reward += delta_num / 50

        return reward

    def evaluate_exploration_rate(self):
        rate = np.sum(self.coverage_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def evaluate_targets_found_rate(self):
        if len(self.target_positions) == 0:
            return 0
        else:
            rate = self.num_targets_found / len(self.target_positions)
            return rate

    def calculate_new_free_area(self):
        old_free_area = self.old_robot_belief == 255
        current_free_area = self.robot_belief == 255

        new_free_area = (current_free_area.astype(np.int) - old_free_area.astype(np.int)) * 255

        return new_free_area, np.sum(old_free_area)

    def calculate_dist_path(self, path):
        dist = 0
        start = path[0]
        end = path[-1]
        for index in path:
            if index == end:
                break
            dist += np.linalg.norm(self.node_coords[start] - self.node_coords[index])
            start = index
        return dist

    def find_frontier(self):
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        # 0-1 unknown area map
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        f = points[ind_to]
        f = f.astype(int)

        f = f * self.resolution

        return f



    def plot_env(self, n, path, step, travel_dist, robots_route, img_path_override=None, sat_path_override=None, msk_name_override=None, sound_id_override=None):

        plt.switch_backend('agg')
        plt.cla()
        color_list = ["r", "g", "c", "m", "y", "k"]

        if not LOAD_AVS_BENCH:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5.5))

        ### Fig: Segmentation Mask ###
        if LOAD_AVS_BENCH:
            ax = ax1
            image = mpimg.imread(img_path_override)
            ax.imshow(image)
            ax.set_title("Ground Image")
            ax.axis("off")

        ### Fig: Environment ###
        msk_name = ""
        if LOAD_AVS_BENCH:
            image = mpimg.imread(sat_path_override)
            msk_name = msk_name_override

            ### Fig1: Environment ###
            ax = ax2
            ax.imshow(image)
            ax.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
            ax.set_title("Image")
            for i, route in enumerate(robots_route):
                robot_marker_color = color_list[i % len(color_list)]
                xPoints = route[0]
                yPoints = route[1]
                ax.plot(xPoints, yPoints, c=robot_marker_color, linewidth=2)
                ax.plot(xPoints[-1], yPoints[-1], markersize=12, zorder=99, marker="^", ls="-", c=robot_marker_color, mec="black")
                ax.plot(xPoints[0], yPoints[0], 'co', c=robot_marker_color, markersize=8, zorder=5)

                # Sensor range
                rng_x = 0.5 * (self.ground_truth.shape[1] / NUM_COORDS_WIDTH)
                rng_y = 0.5 * (self.ground_truth.shape[0] / NUM_COORDS_HEIGHT)
                max_x = min(xPoints[-1] + int(math.ceil(rng_x)), self.ground_truth.shape[1])
                min_x = max(xPoints[-1] - int(math.ceil(rng_x)), 0)
                max_y = min(yPoints[-1] + int(math.ceil(rng_y)), self.ground_truth.shape[0])
                min_y = max(yPoints[-1] - int(math.ceil(rng_y)), 0)
                ax.plot((min_x, min_x), (min_y, max_y), c=robot_marker_color, linewidth=1)
                ax.plot((min_x, max_x), (max_y, max_y), c=robot_marker_color, linewidth=1)
                ax.plot((max_x, max_x), (max_y, min_y), c=robot_marker_color, linewidth=1)
                ax.plot((max_x, min_x), (min_y, min_y), c=robot_marker_color, linewidth=1)


        ### Fig: Graph  ###
        ax = ax3 if LOAD_AVS_BENCH else ax1
        ax.imshow(self.coverage_belief, cmap='gray')
        ax.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        ax.set_title("Information Graph")
        if VIZ_GRAPH_EDGES:
            for i in range(len(self.graph_generator.x)):
                ax.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)
        ax.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=self.filtered_seg_info_mask, zorder=5, s=8)

        for i, route in enumerate(robots_route):
            robot_marker_color = color_list[i % len(color_list)]
            xPoints = route[0]
            yPoints = route[1]
            ax.plot(xPoints, yPoints, c=robot_marker_color, linewidth=2)
            ax.plot(xPoints[-1], yPoints[-1], markersize=12, zorder=99, marker="^", ls="-", c=robot_marker_color, mec="black")
            ax.plot(xPoints[0], yPoints[0], 'co', c=robot_marker_color, markersize=8, zorder=5)

            # Sensor range
            rng_x = 0.5 * (self.ground_truth.shape[1] / NUM_COORDS_WIDTH)
            rng_y = 0.5 * (self.ground_truth.shape[0] / NUM_COORDS_HEIGHT)
            max_x = min(xPoints[-1] + int(math.ceil(rng_x)), self.ground_truth.shape[1])
            min_x = max(xPoints[-1] - int(math.ceil(rng_x)), 0)
            max_y = min(yPoints[-1] + int(math.ceil(rng_y)), self.ground_truth.shape[0])
            min_y = max(yPoints[-1] - int(math.ceil(rng_y)), 0)
            ax.plot((min_x, min_x), (min_y, max_y), c=robot_marker_color, linewidth=1)
            ax.plot((min_x, max_x), (max_y, max_y), c=robot_marker_color, linewidth=1)
            ax.plot((max_x, max_x), (max_y, min_y), c=robot_marker_color, linewidth=1)
            ax.plot((max_x, min_x), (min_y, min_y), c=robot_marker_color, linewidth=1)

        # Plot target positions
        for target in self.target_positions:
            if self.coverage_belief[target[1], target[0]] == 255:
                ax.plot(target[0], target[1], color='g', marker='x', linestyle='-', markersize=12, markeredgewidth=4, zorder=99)
            else:
                ax.plot(target[0], target[1], color='r', marker='x', linestyle='-', markersize=12, markeredgewidth=4, zorder=99)

        ### Fig: Segmentation Mask ###
        ax = ax4 if LOAD_AVS_BENCH else ax2
        if LOAD_AVS_BENCH and USE_CLIP_PREDS:
            H, W = self.ground_truth_size  
            mask_viz = self.segmentation_info_mask.squeeze().reshape((NUM_COORDS_WIDTH, NUM_COORDS_HEIGHT)).T
            im = ax.imshow(
                mask_viz,
                cmap="viridis",
                origin="upper",
                extent=[0, W, H, 0],  
                interpolation="nearest",  
                zorder=0,
            )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_axis_off()  
        else:
            im = ax.imshow(self.segmentation_mask.mean(axis=-1), cmap='viridis', vmin=0, vmax=100)  # cmap='gray'
            ax.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        ax.set_title(f"Predicted Mask (Normalized)")
        for i, route in enumerate(robots_route):
            robot_marker_color = color_list[i % len(color_list)]
            xPoints = route[0]
            yPoints = route[1]
            ax.plot(xPoints, yPoints, c=robot_marker_color, linewidth=2)
            ax.plot(xPoints[-1], yPoints[-1], markersize=12, zorder=99, marker="^", ls="-", c=robot_marker_color, mec="black")
            ax.plot(xPoints[0], yPoints[0], 'co', c=robot_marker_color, markersize=8, zorder=5)

            # Sensor range
            rng_x = 0.5 * (self.ground_truth.shape[1] / NUM_COORDS_WIDTH)
            rng_y = 0.5 * (self.ground_truth.shape[0] / NUM_COORDS_HEIGHT)
            max_x = min(xPoints[-1] + int(math.ceil(rng_x)), self.ground_truth.shape[1])
            min_x = max(xPoints[-1] - int(math.ceil(rng_x)), 0)
            max_y = min(yPoints[-1] + int(math.ceil(rng_y)), self.ground_truth.shape[0])
            min_y = max(yPoints[-1] - int(math.ceil(rng_y)), 0)
            ax.plot((min_x, min_x), (min_y, max_y), c=robot_marker_color, linewidth=1)
            ax.plot((min_x, max_x), (max_y, max_y), c=robot_marker_color, linewidth=1)
            ax.plot((max_x, max_x), (max_y, min_y), c=robot_marker_color, linewidth=1)
            ax.plot((max_x, min_x), (min_y, min_y), c=robot_marker_color, linewidth=1)

        # Add a colorbar 
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Probs")
                    
        if sound_id_override is not None:
            plt.suptitle('Targets Found: {}/{}  Coverage ratio: {:.4g}  Travel Dist: {:.4g} \n ({}) \n (Sound ID: {})'.format(self.num_targets_found, len(self.target_positions), self.explored_rate, travel_dist, msk_name, sound_id_override))
        elif msk_name != "":
            plt.suptitle('Targets Found: {}/{}  Coverage ratio: {:.4g}  Travel Dist: {:.4g} \n ({})'.format(self.num_targets_found, len(self.target_positions), self.explored_rate, travel_dist, msk_name))
        else:
            plt.suptitle('Targets Found: {}/{}  Coverage ratio: {:.4g}  Travel Dist: {:.4g}'.format(self.num_targets_found, len(self.target_positions), self.explored_rate, travel_dist))

        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=100))
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
        plt.close()
