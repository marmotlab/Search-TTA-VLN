#######################################################################
# Name: test_worker.py
#
# - Runs robot in environment using RL Planner
#######################################################################

from .test_parameter import *

import imageio
import os
import copy
import numpy as np
import torch
from time import time
from pathlib import Path
from skimage.transform import resize
from taxabind_avs.satbind.kmeans_clustering import CombinedSilhouetteInertiaClusterer
from .env import Env
from .robot import Robot

np.seterr(invalid='raise', divide='raise')


class TestWorker:
    def __init__(self, meta_agent_id, n_agent, policy_net, global_step, device='cuda', greedy=False, save_image=False, clip_seg_tta=None):
        self.device = device
        self.greedy = greedy
        self.n_agent = n_agent
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image
        self.clip_seg_tta = clip_seg_tta

        self.env = Env(map_index=self.global_step, n_agent=n_agent, k_size=self.k_size, plot=save_image, test=True)
        self.local_policy_net = policy_net

        self.robot_list = []
        self.all_robot_positions = []
        for i in range(self.n_agent):
            robot_position = self.env.start_positions[i]
            robot = Robot(robot_id=i, position=robot_position, plot=save_image)
            self.robot_list.append(robot)
            self.all_robot_positions.append(robot_position)

        self.perf_metrics = dict()
        self.bad_mask_init = False

        if LOAD_AVS_BENCH: 
            if clip_seg_tta is not None: 
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
                print("Chosen k:", self.kmeans_clusterer.final_k)

                if EXECUTE_TTA:
                    print("Will execute TTA...")

        # Define Poisson TTA params
        self.step_since_tta = 0
        self.steps_to_first_tgt = None
        self.steps_to_mid_tgt = None
        self.steps_to_last_tgt = None


    def run_episode(self, curr_episode):

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
        done = False
        for robot_id, deciding_robot in enumerate(self.robot_list):
            deciding_robot.observations = self.get_observations(deciding_robot.robot_position)
            if LOAD_AVS_BENCH and USE_CLIP_PREDS:
                if NUM_COORDS_WIDTH != CLIP_GRIDS_DIMS[0] or NUM_COORDS_HEIGHT != CLIP_GRIDS_DIMS[1]:   # If heatmap is resized from clip original dims
                    heatmap = self.convert_heatmap_resolution(self.clip_seg_tta.heatmap, full_dims=(512, 512), new_dims=(NUM_COORDS_WIDTH, NUM_COORDS_HEIGHT))
                    self.env.segmentation_info_mask = np.expand_dims(heatmap.T.flatten(), axis=1)
                    unnormalized_heatmap = self.convert_heatmap_resolution(self.clip_seg_tta.heatmap_unnormalized, full_dims=(512, 512), new_dims=(NUM_COORDS_WIDTH, NUM_COORDS_HEIGHT))
                    self.env.segmentation_info_mask_unnormalized = np.expand_dims(unnormalized_heatmap.T.flatten(), axis=1)
                    print("Resized heatmap to", NUM_COORDS_WIDTH, "x", NUM_COORDS_HEIGHT)
                else:   
                    self.env.segmentation_info_mask = np.expand_dims(self.clip_seg_tta.heatmap.T.flatten(), axis=1)
                    self.env.segmentation_info_mask_unnormalized = np.expand_dims(self.clip_seg_tta.heatmap_unnormalized.T.flatten(), axis=1)

        ### Run episode ###
        for step in range(NUM_EPS_STEPS):

            next_position_list = []
            dist_list = []
            travel_dist_list = []
            dist_array = np.zeros((self.n_agent, 1))
            for robot_id, deciding_robot in enumerate(self.robot_list):
                observations = deciding_robot.observations

                ### Forward pass through policy to get next position ###
                next_position, action_index = self.select_node(observations)
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
                robot.save_trajectory_coords(self.env.find_index_from_coords(robot.robot_position), self.env.num_new_targets_found)

                # # TTA Update via Poisson Test (with KMeans clustering stats)
                if LOAD_AVS_BENCH and USE_CLIP_PREDS and EXECUTE_TTA:  
                    self.poisson_tta_update(robot, self.global_step, step)

                robot.observations = self.get_observations(robot.robot_position)
                robot.save_reward_done(reward, done)

                # Update metrics
                self.log_metrics(step=step) 

            ### Save a frame to generate gif of robot trajectories ###
            if self.save_image:
                robots_route = []
                for robot in self.robot_list:
                    robots_route.append([robot.xPoints, robot.yPoints])
                if not os.path.exists(GIFS_PATH):
                    os.makedirs(GIFS_PATH)
                if LOAD_AVS_BENCH:
                    sound_id_override = None if self.clip_seg_tta.sound_ids == [] else self.clip_seg_tta.sound_ids[0]
                    self.env.plot_env(
                        self.global_step, 
                        GIFS_PATH, 
                        step, 
                        max(travel_dist_list),
                        robots_route, 
                        img_path_override=self.clip_seg_tta.img_paths[0],    
                        sat_path_override=self.clip_seg_tta.imo_path,
                        msk_name_override=self.clip_seg_tta.species_name, 
                        sound_id_override=sound_id_override
                    )
                else:
                    self.env.plot_env(
                        self.global_step, 
                        GIFS_PATH, 
                        step, 
                        max(travel_dist_list),
                        robots_route
                    )

            if done:
                break

        if LOAD_AVS_BENCH:
            tax = Path(self.clip_seg_tta.gt_mask_name).stem
            self.perf_metrics['tax'] = " ".join(tax.split("_")[1:])
        else:
            self.perf_metrics['tax'] = None
        self.perf_metrics['travel_dist'] = max(travel_dist_list)
        self.perf_metrics['travel_steps'] = step + 1
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
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  
        node_padding_mask = None

        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        bias_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(bias_matrix).float().unsqueeze(0).to(self.device)

        for edges in edge_inputs:
            while len(edges) < self.k_size:
                edges.append(0)        

        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  
        edge_padding_mask = torch.zeros((1, len(edge_inputs), K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations
    

    def select_node(self, observations):
        """ Forward pass through policy to get next position to go to on map """
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask)

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

        # For gif during TTA
        if LOAD_AVS_BENCH:
            with imageio.get_writer('{}/{}_kmeans_stats.gif'.format(path, n), mode='I',
                                    fps=5) as writer:
                for frame in self.kmeans_clusterer.kmeans_frame_files:
                    image = imageio.imread(frame)
                    writer.append_data(image)
            print('Kmeans Clusterer gif complete\n')

            # Remove files
            for filename in self.kmeans_clusterer.kmeans_frame_files[:-1]:
                os.remove(filename)

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

            # # # Adaptative LR (as samples increase, increase LR to fit more datapoints)
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

        # 2) Downsample via your precomputed grid coords
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
