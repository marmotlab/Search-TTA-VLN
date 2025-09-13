#######################################################################
# Name: worker.py
#
# - Runs robot in environment for N steps
# - Collects & Returns S(t), A(t), R(t), S(t+1)
#######################################################################

from .parameter import *

import os
import json
import copy
import imageio
import numpy as np
import torch
from time import time
from .env import Env
from .robot import Robot

class Worker:
    def __init__(self, meta_agent_id, n_agent, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False, clip_seg_tta=None):
        self.device = device
        self.greedy = greedy
        self.n_agent = n_agent
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image
        self.clip_seg_tta = clip_seg_tta

        # Randomize map_index
        mask_index = None
        if MASKS_RAND_INDICES_PATH != "":
            with open(MASKS_RAND_INDICES_PATH, 'r') as f:
                mask_index_rand_json = json.load(f)
            mask_index = mask_index_rand_json[self.global_step % len(mask_index_rand_json)]
            print("mask_index: ", mask_index)

        self.env = Env(map_index=self.global_step, n_agent=n_agent, k_size=self.k_size, plot=save_image, mask_index=mask_index)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.robot_list = []
        self.all_robot_positions = []

        for i in range(self.n_agent):
            robot_position = self.env.start_positions[i]
            robot = Robot(robot_id=i, position=robot_position, plot=save_image)
            self.robot_list.append(robot)
            self.all_robot_positions.append(robot_position)

        self.perf_metrics = dict()
        self.episode_buffer = []
        for i in range(15):
            self.episode_buffer.append([])


    def run_episode(self, curr_episode):

        eps_start = time()
        done = False
        for robot_id, deciding_robot in enumerate(self.robot_list):
            deciding_robot.observations = self.get_observations(deciding_robot.robot_position)

        ### Run episode ###
        for step in range(NUM_EPS_STEPS):

            next_position_list = []
            dist_list = []
            travel_dist_list = []
            dist_array = np.zeros((self.n_agent, 1))
            for robot_id, deciding_robot in enumerate(self.robot_list):
                observations = deciding_robot.observations
                deciding_robot.save_observations(observations)

                ### Forward pass through policy to get next position ###
                next_position, action_index = self.select_node(observations)
                deciding_robot.save_action(action_index)

                dist = np.linalg.norm(next_position - deciding_robot.robot_position)

                ### Log results of action (e.g. distance travelled) ###
                dist_array[robot_id] = dist
                dist_list.append(dist)
                travel_dist_list.append(deciding_robot.travel_dist)
                next_position_list.append(next_position)
                self.all_robot_positions[robot_id] = next_position

            arriving_sequence = np.argsort(dist_list)
            next_position_list = np.array(next_position_list)
            dist_list = np.array(dist_list)
            travel_dist_list = np.array(travel_dist_list)
            next_position_list = next_position_list[arriving_sequence]
            dist_list = dist_list[arriving_sequence]
            travel_dist_list = travel_dist_list[arriving_sequence]

            ### Take Action (Deconflict if 2 agents choose the same target position) ###
            next_position_list, dist_list = self.solve_conflict(arriving_sequence, next_position_list, dist_list)
            reward_list, done = self.env.multi_robot_step(next_position_list, dist_list, travel_dist_list)

            ### Update observations + rewards from action ###
            for reward, robot_id in zip(reward_list, arriving_sequence):
                robot = self.robot_list[robot_id]
                robot.observations = self.get_observations(robot.robot_position)
                robot.save_trajectory_coords(self.env.find_index_from_coords(robot.robot_position), self.env.num_new_targets_found)
                robot.save_reward_done(reward, done)
                robot.save_next_observations(robot.observations)

            ### Save a frame to generate gif of robot trajectories ###
            if self.save_image:
                robots_route = []
                for robot in self.robot_list:
                    robots_route.append([robot.xPoints, robot.yPoints])
                if not os.path.exists(GIFS_PATH):
                    os.makedirs(GIFS_PATH)
                self.env.plot_env(self.global_step, GIFS_PATH, step, max(travel_dist_list), robots_route)
    
            if done:
                break

        for robot in self.robot_list:
            for i in range(15):
                self.episode_buffer[i] += robot.episode_buffer[i]

        self.perf_metrics['travel_dist'] = max(travel_dist_list)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['targets_found'] = self.env.targets_found_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            path = GIFS_PATH
            self.make_gif(path, curr_episode)

        print(YELLOW, f"[Eps {curr_episode} Completed] Time Taken: {time()-eps_start:.2f}s, Steps: {step+1}", NC)


    def get_observations(self, robot_position):
        """ Get robot's sensor observation of environment given position """
        current_node_index = self.env.find_index_from_coords(robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
    
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)
        segmentation_info_mask = copy.deepcopy(self.env.filtered_seg_info_mask)

        n_nodes = node_coords.shape[0]
        node_coords = node_coords / 640
        node_utility = node_utility / 50

        node_utility_inputs = node_utility.reshape((n_nodes, 1))

        occupied_node = np.zeros((n_nodes, 1))
        for position in self.all_robot_positions:
            index = self.env.find_index_from_coords(position)
            if index == current_index.item():
                occupied_node[index] = -1
            else:
                occupied_node[index] = 1

        node_inputs = np.concatenate((node_coords, segmentation_info_mask, guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        bias_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(bias_matrix).float().unsqueeze(0).to(self.device)

        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)
        padding2 = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - len(edge_inputs)))

        for edges in edge_inputs:
            while len(edges) < self.k_size:
                edges.append(0)        

        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, k_size)
        edge_inputs = padding2(edge_inputs)

        edge_padding_mask = torch.zeros((1, len(edge_inputs), K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations


    def select_node(self, observations):
        """ Forward pass through policy to get next position to go to on map """
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[:, current_index.item(), action_index.item()]

        next_position = self.env.node_coords[next_node_index]

        return next_position, action_index


    def solve_conflict(self, arriving_sequence, next_position_list, dist_list):
        """ Deconflict if 2 agents choose the same target position """
        for j, [robot_id, next_position] in enumerate(zip(arriving_sequence, next_position_list)):
            moving_robot = self.robot_list[robot_id]
            # if next_position[0] + next_position[1] * 1j in (next_position_list[:, 0] + next_position_list[:, 1] * 1j)[:j]:
            #     dist_to_next_position = np.argsort(np.linalg.norm(self.env.node_coords - next_position, axis=1))
            #     k = 0
            #     while next_position[0] + next_position[1] * 1j in (next_position_list[:, 0] + next_position_list[:, 1] * 1j)[:j]:
            #         k += 1
            #         next_position = self.env.node_coords[dist_to_next_position[k]]

            dist = np.linalg.norm(next_position - moving_robot.robot_position)
            next_position_list[j] = next_position
            dist_list[j] = dist
            moving_robot.travel_dist += dist
            moving_robot.robot_position = next_position

        return next_position_list, dist_list


    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.run_episode(currEpisode)

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix


    def make_gif(self, path, n):
        """ Generate a gif given list of images """
        with imageio.get_writer('{}/{}_target_rate_{:.2f}.gif'.format(path, n, self.env.targets_found_rate), mode='I',
                                fps=5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)