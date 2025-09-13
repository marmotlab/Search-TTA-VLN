#######################################################################
# Name: robot.py
#
# - Stores S(t), A(t), R(t), S(t+1)
#######################################################################

import torch
from copy import deepcopy

class Robot:
    def __init__(self, robot_id, position, plot=False):
        self.robot_id = robot_id
        self.plot = plot
        self.travel_dist = 0
        self.robot_position = position
        self.observations = None
        self.trajectory_coords = []
        self.targets_found_on_path = []
        
        self.episode_buffer = []
        for i in range(15):
            self.episode_buffer.append([])

        if self.plot:
            self.xPoints = [self.robot_position[0]]
            self.yPoints = [self.robot_position[1]]

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[0] += deepcopy(node_inputs).to('cpu')
        self.episode_buffer[1] += deepcopy(edge_inputs).to('cpu')
        self.episode_buffer[2] += deepcopy(current_index).to('cpu')
        self.episode_buffer[3] += deepcopy(node_padding_mask).to('cpu')
        self.episode_buffer[4] += deepcopy(edge_padding_mask).to('cpu')
        self.episode_buffer[5] += deepcopy(edge_mask).to('cpu')

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += deepcopy(torch.FloatTensor([[[reward]]])).to('cpu')
        self.episode_buffer[8] += deepcopy(torch.tensor([[[(int(done))]]])).to('cpu')
        if self.plot:
            self.xPoints.append(self.robot_position[0])
            self.yPoints.append(self.robot_position[1])

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[9] += deepcopy(node_inputs).to('cpu')
        self.episode_buffer[10] += deepcopy(edge_inputs).to('cpu')
        self.episode_buffer[11] += deepcopy(current_index).to('cpu')
        self.episode_buffer[12] += deepcopy(node_padding_mask).to('cpu')
        self.episode_buffer[13] += deepcopy(edge_padding_mask).to('cpu')
        self.episode_buffer[14] += deepcopy(edge_mask).to('cpu')

    def save_trajectory_coords(self, robot_position_coords, num_target_found):
        self.trajectory_coords.append(robot_position_coords)
        self.targets_found_on_path.append(num_target_found)
