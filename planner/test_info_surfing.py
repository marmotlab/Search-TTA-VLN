#######################################################################
# Name: test_info_surfing.py
#
# - Runs robot in environment using Info Surfing Planner
#######################################################################

import sys
sys.modules['TRAINING'] = False           # False = Inference Testing   

import copy
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from types import SimpleNamespace
from skimage.transform import resize
from taxabind_avs.satbind.kmeans_clustering import CombinedSilhouetteInertiaClusterer
from .env import Env
from .test_parameter import *


OPPOSITE_ACTIONS = {1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
# color
agentColor = (1, 0.2, 0.6)
agentCommColor = (1, 0.6, 0.2)
obstacleColor = (0., 0., 0.)
targetNotFound = (0., 1., 0.)
targetFound = (0.545, 0.27, 0.075)
highestProbColor = (1., 0., 0.)
highestUncertaintyColor = (0., 0., 1.)
lowestProbColor = (1., 1., 1.)


class ISEnv:
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, global_step=0, state=None, shape=(24, 24), numAgents=8, observationSize=11, sensorSize=1, diag=False, save_image=False, clip_seg_tta=None):
        
        self.global_step = global_step
        self.infoMap = None
        self.targetMap = None
        self.agents = []
        self.targets = []
        self.numAgents = numAgents
        self.found_target = []
        self.shape = shape
        self.observationSize = observationSize
        self.sensorSize = sensorSize
        self.diag = diag
        self.communicateCircle = 11
        self.distribs = []
        self.mask = None
        self.finished = False
        self.action_vects = [[-1., 0.], [0., 1.], [1., 0], [0., -1.]] if not diag else [[-1., 0.], [0., 1.], [1., 0], [0., -1.], [-0.707, -0.707], [-0.707, 0.707], [0.707, 0.707], [0.707, -0.707]]
        self.actionlist = []
        self.IS_step = 0
        self.save_image = save_image
        self.clip_seg_tta = clip_seg_tta
        self.perf_metrics = dict()
        self.steps_to_first_tgt = None
        self.steps_to_mid_tgt = None
        self.steps_to_last_tgt = None
        self.targets_found_on_path = []
        self.step_since_tta = 0
        self.IS_frame_files = []
        self.bad_mask_init = False

        # define env
        self.env = Env(map_index=self.global_step, n_agent=numAgents, k_size=K_SIZE, plot=save_image, test=True)
    
        # Overwrite state
        if self.clip_seg_tta is not None:
            self.clip_seg_tta.reset(sample_idx=self.global_step)

            # Override target positions in env
            self.env.target_positions = [(pose[1], pose[0]) for pose in self.clip_seg_tta.target_positions] 

            # Override segmentation mask
            if not USE_CLIP_PREDS and OVERRIDE_MASK_DIR != "":
                score_mask_path = os.path.join(OVERRIDE_MASK_DIR, self.clip_seg_tta.gt_mask_name)
                print("score_mask_path: ", score_mask_path)
                if os.path.exists(score_mask_path):
                    self.env.segmentation_mask = self.env.import_segmentation_mask(score_mask_path)
                    self.env.begin(self.env.map_start_position)
                else:
                    print(f"\n\n{RED}ERROR: Trying to override, but score mask not found at path:{NC} ", score_mask_path)
                    self.bad_mask_init = True

        # Save clustered embeds from sat encoder
        if USE_CLIP_PREDS:
            self.kmeans_clusterer = CombinedSilhouetteInertiaClusterer(
                k_min=1,
                k_max=8,
                k_avg_max=4,
                silhouette_threshold=0.15,
                relative_threshold=0.15,
                random_state=0,
                min_patch_size=5,   
                n_smooth_iter=2,    
                ignore_label=-1,
                plot=self.save_image,
                gifs_dir = GIFS_PATH      
            )    
            # Generate kmeans clusters
            self.kmeans_sat_embeds_clusters = self.kmeans_clusterer.fit_predict(
                patch_embeds=self.clip_seg_tta.patch_embeds,
                map_shape=(CLIP_GRIDS_DIMS[0], CLIP_GRIDS_DIMS[1]),
            )

            if EXECUTE_TTA:
                print("Will execute TTA...")

        IS_info_map = copy.deepcopy(self.env.segmentation_info_mask)
        IS_agent_loc = copy.deepcopy(self.env.start_positions)
        IS_target_loc = copy.deepcopy(self.env.target_positions)
        state=[IS_info_map, IS_agent_loc, IS_target_loc]
        self.setWorld(state)


    def init_render(self):
        """
        Call this once (e.g., in __init__ or just before the scenario loop) 
        to initialize storage for agent paths and turn interactive plotting on.
        """
        # Keep track of each agent's trajectory
        self.trajectories = [[] for _ in range(self.numAgents)]
        self.trajectories_upscaled = [[] for _ in range(self.numAgents)]
        
        # Turn on interactive mode so we can update the same figure repeatedly
        plt.ion()
        plt.figure(figsize=(6,6))
        plt.title("Information Map with Agents, Targets, and Sensor Ranges")


    def record_positions(self):
        """
        Call this after all agents have moved in a step (or whenever you want to update 
        the trajectory). It appends the current positions of each agent to `self.trajectories`.
        """
        for idx, agent in enumerate(self.agents):
            self.trajectories[idx].append((agent.row, agent.col))
            self.trajectories_upscaled[idx].append(self.env.graph_generator.grid_coords[agent.row, agent.col])


    def render(self, episode_num, step_num):
        """
        Renders the current state in a single matplotlib plot.
        Ensures consistent image size for GIF generation.
        """
        
        # Completely reset the figure to avoid leftover state
        plt.close('all')
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        ax = fig.add_subplot(111)

        # Plot the information map
        ax.imshow(self.infoMap, origin='lower', cmap='gray')

        # Show agent positions and their trajectories
        for idx, agent in enumerate(self.agents):
            positions = self.trajectories[idx]
            if len(positions) > 1:
                rows = [p[0] for p in positions]
                cols = [p[1] for p in positions]
                ax.plot(cols, rows, linewidth=1)

            ax.scatter(agent.col, agent.row, marker='o', s=50)

        # Plot target locations
        for t in self.targets:
            color = 'green' if np.isnan(t.time_found) else 'red'
            ax.scatter(t.col, t.row, marker='x', s=100, color=color)

        # Title and axis formatting
        ax.set_title(f"Step: {self.IS_step}")
        ax.invert_yaxis()

        # Create output folder if it doesn't exist
        if not os.path.exists(GIFS_PATH):
            os.makedirs(GIFS_PATH)

        # Save the frame with consistent canvas
        frame_path = f'{GIFS_PATH}/IS_{episode_num}_{step_num}.png'
        plt.savefig(frame_path, bbox_inches='tight', pad_inches=0.1)
        self.IS_frame_files.append(frame_path)

        # Cleanup
        plt.close(fig)


    def setWorld(self, state=None):
        """
        1. empty all the element
        2. create the new episode
        """
        if state is not None:
            self.infoMap = copy.deepcopy(state[0].reshape(self.shape).T)
            agents = []
            self.numAgents = len(state[1])
            for a in range(1, self.numAgents + 1):
                abs_pos = state[1].pop(0)
                abs_pos = np.array(abs_pos)
                row, col = self.env.graph_generator.find_closest_index_from_grid_coords_2d(np.array(abs_pos))
                agents.append(Agent(ID=a, row=row, col=col, sensorSize=self.sensorSize, infoMap=np.copy(self.infoMap),
                                    uncertaintyMap=np.copy(self.infoMap), shape=self.shape, numAgents=self.numAgents))
            self.agents = agents

            targets, n_targets = [], 1
            for t in range(len(state[2])):
                abs_pos = state[2].pop(0)
                abs_pos = np.array(abs_pos)
                row, col = self.env.graph_generator.find_closest_index_from_grid_coords_2d(abs_pos)
                targets.append(Target(ID=n_targets, row=row, col=col, time_found=np.nan))
                n_targets = n_targets + 1
            self.targets = targets

    def extractObservation(self, agent):
        """
        Extract observations from information map
        """

        transform_row = self.observationSize // 2 - agent.row
        transform_col = self.observationSize // 2 - agent.col

        observation_layers = np.zeros((1, self.observationSize, self.observationSize))
        min_row = max((agent.row - self.observationSize // 2), 0)
        max_row = min((agent.row + self.observationSize // 2 + 1), self.shape[0])
        min_col = max((agent.col - self.observationSize // 2), 0)
        max_col = min((agent.col + self.observationSize // 2 + 1), self.shape[1])

        observation = np.full((self.observationSize, self.observationSize), 0.)
        infoMap = np.full((self.observationSize, self.observationSize), 0.)
        densityMap = np.full((self.observationSize, self.observationSize), 0.)

        infoMap[(min_row + transform_row):(max_row + transform_row),
                    (min_col + transform_col):(max_col + transform_col)] = self.infoMap[
                                                                           min_row:max_row, min_col:max_col]
        observation_layers[0] = infoMap

        return observation_layers


    def listNextValidActions(self, agent_id, prev_action=0):
        """
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        """
        available_actions = [0]
        agent = self.agents[agent_id - 1]  

        MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]  
        size = 4 + self.diag * 4
        for action in range(size):
            out_of_bounds = agent.row + MOVES[action][0] >= self.shape[0] \
                            or agent.row + MOVES[action][0] < 0\
                            or agent.col + MOVES[action][1] >= self.shape[1] \
                            or agent.col + MOVES[action][1] < 0

            if (not out_of_bounds) and not (prev_action == OPPOSITE_ACTIONS[action + 1]):
                available_actions.append(action + 1)

        return np.array(available_actions)


    def executeAction(self, agentID, action, timeStep):
        """
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        LeftUp (-1,-1) : 5
        RightUP (-1,1) :6
        RightDown (1,1) :7
        RightLeft (1,-1) :8
        """
        agent = self.agents[agentID - 1]
        origLoc = agent.getLocation()

        if (action >= 1) and (action <= 8):
            agent.move(action)
            row, col = agent.getLocation()

            # If the move is not valid, roll it back
            if (row < 0) or (col < 0) or (row >= self.shape[0]) or (col >= self.shape[1]):
                self.updateInfoCheckTarget(agentID, timeStep, origLoc)
                return 0

        elif action == 0:
            self.updateInfoCheckTarget(agentID, timeStep, origLoc)
            return 0

        else:
            print("INVALID ACTION: {}".format(action))
            sys.exit()

        newLoc = agent.getLocation()
        self.updateInfoCheckTarget(agentID, timeStep, origLoc)
        return action


    def updateInfoCheckTarget(self, agentID, timeStep, origLoc):
        """
        update the self.infoMap and check whether the agent has found a target
        """
        agent = self.agents[agentID - 1]
        transform_row = self.sensorSize // 2 - agent.row
        transform_col = self.sensorSize // 2 - agent.col

        min_row = max((agent.row - self.sensorSize // 2), 0)
        max_row = min((agent.row + self.sensorSize // 2 + 1), self.shape[0])
        min_col = max((agent.col - self.sensorSize // 2), 0)
        max_col = min((agent.col + self.sensorSize // 2 + 1), self.shape[1])
        for t in self.targets:
            if (t.row == agent.row) and (t.col == agent.col):
                t.updateFound(timeStep)
                self.found_target.append(t)
                t.status = True

        self.infoMap[min_row:max_row, min_col:max_col] *= 0.05  


    def updateInfoEntireTrajectory(self, agentID):
        """
        update the self.infoMap and check whether the agent has found a target
        """
        traj = self.trajectories[agentID - 1]

        for (row,col) in traj:
            min_row = max((row - self.sensorSize // 2), 0)
            max_row = min((row + self.sensorSize // 2 + 1), self.shape[0])
            min_col = max((col - self.sensorSize // 2), 0)
            max_col = min((col + self.sensorSize // 2 + 1), self.shape[1])
            self.infoMap[min_row:max_row, min_col:max_col] *= 0.05  


    # Execute one time step within the environment
    def step(self, agentID, action, timeStep):
        """
        the agents execute the actions
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        """
        assert (agentID > 0)

        self.executeAction(agentID, action, timeStep)


    def observe(self, agentID):
        assert (agentID > 0)
        vectorObs = self.extractObservation(self.agents[agentID - 1])
        return [vectorObs]


    def check_finish(self):
        if TERMINATE_ON_TGTS_FOUND:
            found_status = [t.time_found for t in self.targets]
            d = False
            if np.isnan(found_status).sum() == 0:
                d = True
            return d
        else:
            return False    


    def gradVec(self, observation, agent):
        a = observation[0]  

        # Make info & unc cells with low value as 0
        a[a < 0.0002] = 0.0

        # Center square from 11x11
        a_11x11 = a[4:7, 4:7]
        m_11x11 = np.array((a_11x11))

        # Center square from 9x9
        a_9x9 = self.pooling(a, (3, 3), stride=(1, 1), method='max', pad=False)    
        a_9x9 = a_9x9[3:6, 3:6]
        m_9x9 = np.array((a_9x9))

        # Center square from 6x6
        a_6x6 = self.pooling(a, (6, 6), stride=(1, 1), method='max', pad=False)    
        a_6x6 = a_6x6[1:4, 1:4]
        m_6x6 = np.array((a_6x6))

        # Center square from 3x3
        a_3x3 = self.pooling(a, (5, 5), stride=(3, 3), method='max', pad=False)
        m_3x3 = np.array((a_3x3))

        # Merging multiScales with weights
        m = m_3x3 * 0.25 + m_6x6 * 0.25 + m_9x9 * 0.25 + m_11x11 * 0.25
        a = m

        adx, ady = np.gradient(a)
        den = np.linalg.norm(np.array([adx[1, 1], ady[1, 1]]))
        if (den != 0) and (not np.isnan(den)):
            infovec = np.array([adx[1, 1], ady[1, 1]]) / den
        else:
            infovec = 0
        agentvec = []

        if len(agentvec) == 0:
            den = np.linalg.norm(infovec)
            if (den != 0) and (not np.isnan(den)):
                direction = infovec / den
            else:
                direction = self.action_vects[np.random.randint(4 + self.diag * 4)]
        else:
            den = np.linalg.norm(np.mean(agentvec, 0))
            if (den != 0) and (not np.isnan(den)):
                agentvec = np.mean(agentvec, 0) / den
            else:
                agentvec = 0

            den = np.linalg.norm(0.6 * infovec + 0.4 * agentvec)
            if (den != 0) and (not np.isnan(den)):
                direction = (0.6 * infovec + 0.4 * agentvec) / den
            else:
                direction = self.action_vects[np.random.randint(4 + self.diag * 4)]

        action_vec = [[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]] if not self.diag else [[0., 0.], [-1., 0.], [0., 1.], [1., 0], [0., -1.], [-0.707, -0.707], [-0.707, 0.707], [0.707, 0.707], [0.707, -0.707]]
        actionid = np.argmax([np.dot(direction, a) for a in action_vec])
        actionid = self.best_valid_action(actionid, agent, direction)
        return actionid


    def best_valid_action(self, actionid, agent, direction):
        if len(self.actionlist) > 1:
            if self.action_invalid(actionid, agent):
                action_vec = [[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]] if not self.diag else [[0., 0.], [-1., 0.], [0., 1.], [1., 0], [0., -1.], [-0.707, -0.707], [-0.707, 0.707], [0.707, 0.707], [0.707, -0.707]]
                actionid = np.array([np.dot(direction, a) for a in action_vec])
                actionid = actionid.argsort()
                pi = 3 + self.diag*4 
                while self.action_invalid(actionid[pi], agent) and pi >= 0:
                    pi -= 1
                if pi == -1:
                    return OPPOSITE_ACTIONS[self.actionlist[self.IS_step - 1][agent - 1]]
                elif actionid[pi] == 0:
                    return OPPOSITE_ACTIONS[self.actionlist[self.IS_step - 1][agent - 1]]
                else:
                    return actionid[pi]
        return actionid


    def action_invalid(self, action, agent):
        # Going back to the previous cell is disabled
        if action == OPPOSITE_ACTIONS[self.actionlist[self.IS_step - 1][agent - 1]]:
            return True
        # Move N,E,S,W
        if (action >= 1) and (action <= 8):
            agent = self.agents[agent - 1]
            agent.move(action)
            row, col = agent.getLocation()

            # If the move is not valid, roll it back
            if ((row < 0) or (col < 0) or (row >= self.shape[0]) or (col >= self.shape[1])):
                agent.reverseMove(action)
                return True

            agent.reverseMove(action)
            return False
        return False


    def step_all_parallel(self):
        actions = []
        reward = 0
        # Decide actions for each agent
        for agent_id in range(1, self.numAgents + 1):
            o = self.observe(agent_id)
            actions.append(self.gradVec(o[0], agent_id))
        self.actionlist.append(actions)

        # Execute those actions
        for agent_id in range(1, self.numAgents + 1):
            self.step(agent_id, actions[agent_id - 1], self.IS_step)

        # Record for visualization
        self.record_positions()

    def is_scenario(self, max_step=512, episode_number=0):

        # Return all metrics as None if faulty mask init
        if self.bad_mask_init:
            self.perf_metrics['tax'] = None
            self.perf_metrics['travel_dist'] = None
            self.perf_metrics['travel_steps'] = None
            self.perf_metrics['steps_to_first_tgt'] = None
            self.perf_metrics['steps_to_mid_tgt'] = None
            self.perf_metrics['steps_to_last_tgt'] = None
            self.perf_metrics['explored_rate'] = None
            self.perf_metrics['targets_found'] = None
            self.perf_metrics['targets_total'] = None
            self.perf_metrics['kmeans_k'] = None
            self.perf_metrics['tgts_gt_score'] = None
            self.perf_metrics['clip_inference_time'] = None
            self.perf_metrics['tta_time'] = None
            self.perf_metrics['success_rate'] = None
            return

        eps_start = time()
        self.IS_step = 0
        self.finished = False
        reward = 0

        # Initialize the rendering just once before the loop
        self.init_render()
        self.record_positions()

        # Initial Setup
        if LOAD_AVS_BENCH and USE_CLIP_PREDS:
            if NUM_COORDS_WIDTH != CLIP_GRIDS_DIMS[0] or NUM_COORDS_HEIGHT != CLIP_GRIDS_DIMS[1]:   # If heatmap is resized from clip original dims
                heatmap = self.convert_heatmap_resolution(self.clip_seg_tta.heatmap, full_dims=(512, 512), new_dims=(NUM_COORDS_WIDTH, NUM_COORDS_HEIGHT))
                self.env.segmentation_info_mask = np.expand_dims(heatmap.T.flatten(), axis=1)
                unnormalized_heatmap = self.convert_heatmap_resolution(self.clip_seg_tta.heatmap_unnormalized, full_dims=(512, 512), new_dims=(NUM_COORDS_WIDTH, NUM_COORDS_HEIGHT))
                self.env.segmentation_info_mask_unnormalized = np.expand_dims(unnormalized_heatmap.T.flatten(), axis=1)
                self.infoMap = copy.deepcopy(heatmap) 
                print("Resized heatmap to", NUM_COORDS_WIDTH, "x", NUM_COORDS_HEIGHT)
            else:   
                self.env.segmentation_info_mask = np.expand_dims(self.clip_seg_tta.heatmap.T.flatten(), axis=1)
                self.env.segmentation_info_mask_unnormalized = np.expand_dims(self.clip_seg_tta.heatmap_unnormalized.T.flatten(), axis=1)
                self.infoMap = copy.deepcopy(self.clip_seg_tta.heatmap) 

        self.targets_found_on_path.append(self.env.num_new_targets_found)

        while self.IS_step < max_step and not self.check_finish():
            self.step_all_parallel()
            self.IS_step += 1

            # Render after each step
            if self.save_image:
                self.render(episode_num=self.global_step, step_num=self.IS_step)

            # Update in env
            next_position_list = [self.trajectories_upscaled[i][-1] for i, agent in enumerate(self.agents)]
            dist_list = [0 for _ in range(self.numAgents)]  
            travel_dist_list = [self.compute_travel_distance(traj) for traj in self.trajectories]   
            self.env.multi_robot_step(next_position_list, dist_list, travel_dist_list)
            self.targets_found_on_path.append(self.env.num_new_targets_found)

            # TTA Update via Poisson Test (with KMeans clustering stats)
            robot_id = 0     # Assume 1 agent for now
            robot_traj = self.trajectories[robot_id]
            if LOAD_AVS_BENCH and USE_CLIP_PREDS and EXECUTE_TTA:  
                flat_traj_coords = [robot_traj[i][1] * self.shape[0] + robot_traj[i][0] for i in range(len(robot_traj))]
                robot = SimpleNamespace(
                    trajectory_coords=flat_traj_coords,
                    targets_found_on_path=self.targets_found_on_path
                )
                self.poisson_tta_update(robot, self.global_step, self.IS_step)
                self.infoMap = copy.deepcopy(self.env.segmentation_info_mask.reshape((self.shape[1],self.shape[0])).T) 
                self.updateInfoEntireTrajectory(robot_id)

            # Update metrics
            self.log_metrics(step=self.IS_step-1)

            ### Save a frame to generate gif of robot trajectories ###
            if self.save_image:
                robots_route = [ ([], []) ] # Assume 1 robot
                for point in self.trajectories_upscaled[robot_id]:
                    robots_route[robot_id][0].append(point[0])
                    robots_route[robot_id][1].append(point[1])
                if not os.path.exists(GIFS_PATH):
                    os.makedirs(GIFS_PATH)
                if LOAD_AVS_BENCH:
                    sound_id_override = None if self.clip_seg_tta.sound_ids == [] else self.clip_seg_tta.sound_ids[0]
                    self.env.plot_env(
                        self.global_step, 
                        GIFS_PATH, 
                        self.IS_step-1, 
                        max(travel_dist_list),
                        robots_route, 
                        img_path_override=self.clip_seg_tta.img_paths[0],  # Viz 1st 
                        sat_path_override=self.clip_seg_tta.imo_path,
                        msk_name_override=self.clip_seg_tta.species_name, 
                        sound_id_override=sound_id_override,
                    )
                else:
                    self.env.plot_env(
                        self.global_step, 
                        GIFS_PATH, 
                        self.IS_step-1, 
                        max(travel_dist_list),
                        robots_route
                    )

        # Log metrics
        if LOAD_AVS_BENCH:
            tax = Path(self.clip_seg_tta.gt_mask_name).stem
            self.perf_metrics['tax'] = " ".join(tax.split("_")[1:])
        else:
            self.perf_metrics['tax'] = None
        travel_distances = [self.compute_travel_distance(traj) for traj in self.trajectories]
        self.perf_metrics['travel_dist'] = max(travel_distances)
        self.perf_metrics['travel_steps'] = self.IS_step
        self.perf_metrics['steps_to_first_tgt'] = self.steps_to_first_tgt
        self.perf_metrics['steps_to_mid_tgt'] = self.steps_to_mid_tgt
        self.perf_metrics['steps_to_last_tgt'] = self.steps_to_last_tgt
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['targets_found'] = self.env.targets_found_rate
        self.perf_metrics['targets_total'] = len(self.env.target_positions)
        if USE_CLIP_PREDS:
            self.perf_metrics['kmeans_k'] = self.kmeans_clusterer.final_k
            self.perf_metrics['tgts_gt_score'] = self.clip_seg_tta.tgts_gt_score
            self.perf_metrics['clip_inference_time'] = self.clip_seg_tta.clip_inference_time
            self.perf_metrics['tta_time'] = self.clip_seg_tta.tta_time
        else:
            self.perf_metrics['kmeans_k'] = None
            self.perf_metrics['tgts_gt_score'] = None
            self.perf_metrics['clip_inference_time'] = None
            self.perf_metrics['tta_time'] = None
        if FORCE_LOGGING_DONE_TGTS_FOUND and self.env.targets_found_rate == 1.0:
            self.perf_metrics['success_rate'] = True
        else:
            self.perf_metrics['success_rate'] = self.env.check_done()[0]

        # save gif
        if self.save_image:
            path = GIFS_PATH
            self.make_gif(path, self.global_step)

        print(YELLOW, f"[Eps {episode_number} Completed] Time Taken: {time()-eps_start:.2f}s, Steps: {self.IS_step}", NC)


    def asStride(self, arr, sub_shape, stride):
        """
        Get a strided sub-matrices view of an ndarray.
        See also skimage.util.shape.view_as_windows()
        """
        s0, s1 = arr.strides[:2]
        m1, n1 = arr.shape[:2]
        m2, n2 = sub_shape
        view_shape = (1+(m1-m2)//stride[0], 1+(n1-n2)//stride[1], m2, n2)+arr.shape[2:]
        strides = (stride[0]*s0, stride[1]*s1, s0, s1)+arr.strides[2:]
        subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
        return subs


    def pooling(self, mat, ksize, stride=None, method='max', pad=False):
        """
        Overlapping pooling on 2D or 3D data.

        <mat>: ndarray, input array to pool.
        <ksize>: tuple of 2, kernel size in (ky, kx).
        <stride>: tuple of 2 or None, stride of pooling window.
                  If None, same as <ksize> (non-overlapping pooling).
        <method>: str, 'max for max-pooling,
                       'mean' for mean-pooling.
        <pad>: bool, pad <mat> or not. If no pad, output has size
               (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
               if pad, output has size ceil(n/s).

        Return <result>: pooled matrix.
        """

        m, n = mat.shape[:2]
        ky, kx = ksize
        if stride is None:
            stride = (ky, kx)
        sy, sx = stride

        _ceil = lambda x, y: int(np.ceil(x/float(y)))

        if pad:
            ny = _ceil(m,sy)
            nx = _ceil(n,sx)
            size = ((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
            mat_pad = np.full(size,np.nan)
            mat_pad[:m,:n,...] = mat
        else:
            mat_pad = mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

        view = self.asStride(mat_pad,ksize,stride)

        if method == 'max':
            result = np.nanmax(view,axis=(2,3))
        else:
            result = np.nanmean(view,axis=(2,3))

        return result


    def compute_travel_distance(self, trajectory):
        distance = 0.0
        for i in range(1, len(trajectory)):
            # Convert the tuple positions to numpy arrays for easy computation.
            prev_pos = np.array(trajectory[i-1])
            curr_pos = np.array(trajectory[i])
            # Euclidean distance between consecutive positions.
            distance += np.linalg.norm(curr_pos - prev_pos)
        return distance

    ################################################################################
    # SPPP Related Fns
    ################################################################################

    def log_metrics(self, step):
        # Update tgt found metrics
        if self.steps_to_first_tgt is None and self.env.num_targets_found == 1:
            self.steps_to_first_tgt = step + 1
        if self.steps_to_mid_tgt is None and self.env.num_targets_found == int(len(self.env.target_positions) / 2):
            self.steps_to_mid_tgt = step + 1
        if self.steps_to_last_tgt is None and self.env.num_targets_found == len(self.env.target_positions):
            self.steps_to_last_tgt = step + 1


    def transpose_flat_idx(self, idx, H=NUM_COORDS_HEIGHT, W=NUM_COORDS_WIDTH):
        """
        Transpose a flat index from an ``H×W`` grid to the equivalent
        position in the ``W×H`` transposed grid while **keeping the result
        in 1-D**.
        """
        # --- Safety check to catch out-of-range indices ---
        assert 0 <= idx < H * W, f"idx {idx} out of bounds for shape ({H}, {W})"

        # Original (row, col)
        row, col = divmod(idx, W)
        # After transpose these coordinates swap
        row_T, col_T = col, row

        # Flatten back into 1-D (row-major) for the W×H grid
        return row_T * H + col_T
    

    def poisson_tta_update(self, robot, episode, step):

        # Generate Kmeans Clusters Stats
        # Scale index back to CLIP_GRIDS_DIMS to be compatible with CLIP patch size
        if NUM_COORDS_WIDTH != CLIP_GRIDS_DIMS[0] or NUM_COORDS_HEIGHT != CLIP_GRIDS_DIMS[1]:
            # High-res remap via pixel coordinates preserves exact neighbourhood
            filt_traj_coords, filt_targets_found_on_path = self.scale_trajectory(
                robot.trajectory_coords,
                self.env.target_positions,
                old_dims=(NUM_COORDS_HEIGHT, NUM_COORDS_WIDTH),
                full_dims=(512, 512),
                new_dims=(CLIP_GRIDS_DIMS[0], CLIP_GRIDS_DIMS[1])
            )
        else:
            filt_traj_coords = [self.transpose_flat_idx(idx) for idx in robot.trajectory_coords]
            filt_targets_found_on_path = robot.targets_found_on_path

        region_stats_dict = self.kmeans_clusterer.compute_region_statistics(
            self.kmeans_sat_embeds_clusters,
            self.clip_seg_tta.heatmap_unnormalized,
            filt_traj_coords,
            episode_num=episode,
            step_num=step
        )

        # Prep & execute TTA
        self.step_since_tta += 1
        if robot.targets_found_on_path[-1] or self.step_since_tta % STEPS_PER_TTA == 0:  
                
            num_cells = self.clip_seg_tta.heatmap.shape[0] * self.clip_seg_tta.heatmap.shape[1]
            pos_sample_weight_scale, neg_sample_weight_scale = [], []

            for i, sample_loc in enumerate(filt_traj_coords):
                label = self.kmeans_clusterer.get_label_id(self.kmeans_sat_embeds_clusters, sample_loc)
                num_patches = region_stats_dict[label]['num_patches']
                patches_visited = region_stats_dict[label]['patches_visited']
                expectation = region_stats_dict[label]['expectation']

                # Exponent like focal loss to wait for more samples before confidently decreasing
                pos_weight = 4.0 
                neg_weight = min(1.0, (patches_visited/(3*num_patches))**GAMMA_EXPONENT)  
                pos_sample_weight_scale.append(pos_weight)
                neg_sample_weight_scale.append(neg_weight)

            # Adaptative LR (as samples increase, increase LR to fit more datapoints)
            adaptive_lr = MIN_LR + (MAX_LR - MIN_LR) * (step / num_cells)
                
            # TTA Update
            self.clip_seg_tta.execute_tta(
                filt_traj_coords, 
                filt_targets_found_on_path, 
                tta_steps=NUM_TTA_STEPS, 
                lr=adaptive_lr,
                pos_sample_weight=pos_sample_weight_scale,
                neg_sample_weight=neg_sample_weight_scale,
                reset_weights=RESET_WEIGHTS
            )
            if NUM_COORDS_WIDTH != CLIP_GRIDS_DIMS[0] or NUM_COORDS_HEIGHT != CLIP_GRIDS_DIMS[1]:   # If heatmap is resized from clip original dims
                heatmap = self.convert_heatmap_resolution(self.clip_seg_tta.heatmap, full_dims=(512, 512), new_dims=(NUM_COORDS_WIDTH, NUM_COORDS_HEIGHT))
                self.env.segmentation_info_mask = np.expand_dims(heatmap.T.flatten(), axis=1)
                unnormalized_heatmap = self.convert_heatmap_resolution(self.clip_seg_tta.heatmap_unnormalized, full_dims=(512, 512), new_dims=(NUM_COORDS_WIDTH, NUM_COORDS_HEIGHT))
                self.env.segmentation_info_mask_unnormalized = np.expand_dims(unnormalized_heatmap.T.flatten(), axis=1)
                print("~Resized heatmap to", NUM_COORDS_WIDTH, "x", NUM_COORDS_HEIGHT)
            else:  
                self.env.segmentation_info_mask = np.expand_dims(self.clip_seg_tta.heatmap.T.flatten(), axis=1)
                self.env.segmentation_info_mask_unnormalized = np.expand_dims(self.clip_seg_tta.heatmap_unnormalized.T.flatten(), axis=1)

            self.step_since_tta = 0


    def convert_heatmap_resolution(self, heatmap, full_dims=(512, 512), new_dims=(24, 24)):
        heatmap_large = resize(heatmap, full_dims, order=1,  # order=1 → bilinear
             mode='reflect', anti_aliasing=True)
        
        coords = self.env.graph_generator.grid_coords   # (N, N, 2)
        rows, cols = coords[...,1], coords[...,0]
        heatmap_resized = heatmap_large[rows, cols]
        heatmap_resized = heatmap_resized.reshape(new_dims[1], new_dims[0])
        return heatmap_resized


    def convert_labelmap_resolution(self, labelmap, full_dims=(512, 512), new_dims=(24, 24)):
        """
        1) Upsample via nearest‐neighbor to full_dims
        2) Sample back down to your graph grid using grid_coords
        """
        # 1) Upsample with nearest‐neighbor, preserving integer labels
        up = resize(
            labelmap,
            full_dims,
            order=0,                # nearest‐neighbor
            mode='edge',            # padding mode
            preserve_range=True,    # don't normalize labels
            anti_aliasing=False     # must be False for labels
        ).astype(labelmap.dtype)    # back to original integer dtype

        # 2) Downsample via your precomputed grid coords (N×N×2)
        coords = self.env.graph_generator.grid_coords   # shape (N, N, 2)
        rows = coords[...,1].astype(int)
        cols = coords[...,0].astype(int)

        small = up[rows, cols]      # shape (N, N)
        small = small.reshape(new_dims[0], new_dims[1])
        return small
    

    def scale_trajectory(self,
                        flat_indices,
                        targets,
                        old_dims=(17, 17),
                        full_dims=(512, 512),
                        new_dims=(24, 24)):
        """
        Args:
            flat_indices: list of ints in [0..old_H*old_W-1]
            targets:      list of (y_pix, x_pix) in [0..full_H-1]
            old_dims:     (old_H, old_W)
            full_dims:    (full_H, full_W)
            new_dims:     (new_H, new_W)

        Returns:
            new_flat_traj: list of unique flattened indices in new_H×new_W
            counts:        list of ints, same length as new_flat_traj
        """
        old_H, old_W = old_dims
        full_H, full_W = full_dims
        new_H,  new_W  = new_dims

        # 1) bin targets into new grid
        cell_h_new = full_H / new_H
        cell_w_new = full_W / new_W
        grid_counts = [[0]*new_W for _ in range(new_H)]
        for x_pix, y_pix in targets:  # note (x, y) order as in original implementation
            i_t = min(int(y_pix / cell_h_new), new_H - 1)
            j_t = min(int(x_pix / cell_w_new), new_W - 1)
            grid_counts[i_t][j_t] += 1

        # 2) Walk the trajectory indices and project each old cell's *entire
        #    pixel footprint* onto the finer 24×24 grid.
        cell_h_full = full_H / old_H
        cell_w_full = full_W / old_W

        seen = set()
        new_flat_traj = []

        for node_idx in flat_indices:
            if node_idx < 0 or node_idx >= len(self.env.graph_generator.node_coords):
                continue

            coord_xy = self.env.graph_generator.node_coords[node_idx]
            try:
                row_old, col_old = self.env.graph_generator.find_index_from_grid_coords_2d(coord_xy)
            except Exception:
                continue

            # Bounding box of the old cell in full-resolution pixel space
            y0 = row_old * cell_h_full
            y1 = (row_old + 1) * cell_h_full
            x0 = col_old * cell_w_full
            x1 = (col_old + 1) * cell_w_full

            # Which new-grid rows & cols overlap? (inclusive ranges)
            i_start = max(0, min(int(y0 / cell_h_new), new_H - 1))
            i_end   = max(0, min(int((y1 - 1) / cell_h_new), new_H - 1))
            j_start = max(0, min(int(x0 / cell_w_new), new_W - 1))
            j_end   = max(0, min(int((x1 - 1) / cell_w_new), new_W - 1))

            for ii in range(i_start, i_end + 1):
                for jj in range(j_start, j_end + 1):
                    f_new = ii * new_W + jj
                    if f_new not in seen:
                        seen.add(f_new)
                        new_flat_traj.append(f_new)

        # 3) annotate counts
        counts = []
        for f in new_flat_traj:
            i_new, j_new = divmod(f, new_W)
            counts.append(grid_counts[i_new][j_new])

        return new_flat_traj, counts


    ################################################################################

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

        # For KMeans gif 
        if LOAD_AVS_BENCH and USE_CLIP_PREDS:
            with imageio.get_writer('{}/{}_kmeans_stats.gif'.format(path, n), mode='I',
                                    fps=5) as writer:
                for frame in self.kmeans_clusterer.kmeans_frame_files:
                    image = imageio.imread(frame)
                    writer.append_data(image)
            print('Kmeans Clusterer gif complete\n')

            # Remove files
            for filename in self.kmeans_clusterer.kmeans_frame_files[:-1]:
                os.remove(filename)


        # IS gif
        with imageio.get_writer('{}/{}_IS.gif'.format(path, n), mode='I',
                                fps=5) as writer:
            for frame in self.IS_frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('Kmeans Clusterer gif complete\n')

        # Remove files
        for filename in self.IS_frame_files[:-1]:
            os.remove(filename)

    ################################################################################


class Agent:
    def __init__(self, ID, infoMap=None, uncertaintyMap=None, shape=None, row=0, col=0, sensorSize=9, numAgents=8):
        self.ID = ID
        self.row = row
        self.col = col
        self.numAgents = numAgents
        self.sensorSize = sensorSize

    def setLocation(self, row, col):
        self.row = row
        self.col = col

    def getLocation(self):
        return [self.row, self.col]

    def move(self, action):
        """
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        LeftUp (-1,-1) : 5
        RightUP (-1,1) :6
        RightDown (1,1) :7
        RightLeft (1,-1) :8
        check valid action of the agent. be sure not to be out of the boundary
        """
        if action == 0:
            return 0
        elif action == 1:
            self.row -= 1
        elif action == 2:
            self.col += 1
        elif action == 3:
            self.row += 1
        elif action == 4:
            self.col -= 1
        elif action == 5:
            self.row -= 1
            self.col -= 1
        elif action == 6:
            self.row -= 1
            self.col += 1
        elif action == 7:
            self.row += 1
            self.col += 1
        elif action == 8:
            self.row += 1
            self.col -= 1

    def reverseMove(self, action):
        if action == 0:
            return 0
        elif action == 1:
            self.row += 1
        elif action == 2:
            self.col -= 1
        elif action == 3:
            self.row -= 1
        elif action == 4:
            self.col += 1
        elif action == 5:
            self.row += 1
            self.col += 1
        elif action == 6:
            self.row += 1
            self.col -= 1
        elif action == 7:
            self.row -= 1
            self.col -= 1
        elif action == 8:
            self.row -= 1
            self.col += 1
        else:
            print("agent can only move NESW/1234")
            sys.exit()


class Target:
    def __init__(self, row, col, ID, time_found=np.nan):
        self.row = row
        self.col = col
        self.ID = ID
        self.time_found = time_found
        self.status = None
        self.time_visited = time_found

    def getLocation(self):
        return self.row, self.col

    def updateFound(self, timeStep):
        if np.isnan(self.time_found):
            self.time_found = timeStep

    def updateVisited(self, timeStep):
        if np.isnan(self.time_visited):
            self.time_visited = timeStep


if __name__ == "__main__":

    search_env = Env(map_index=1, k_size=K_SIZE, n_agent=NUM_ROBOTS, plot=SAVE_GIFS)

    IS_info_map = search_env.segmentation_info_mask
    IS_agent_loc = search_env.start_positions
    IS_target_loc = [[312, 123], [123, 312], [312, 312], [123, 123]]

    env = ISEnv(state=[IS_info_map, IS_agent_loc, IS_target_loc], shape=(NUM_COORDS_HEIGHT, NUM_COORDS_WIDTH))
    env.is_scenario(NUM_EPS_STEPS)
    print()
