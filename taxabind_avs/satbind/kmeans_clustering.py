##############################################################################
# Name: kmeans_clustering.py
#
# - Performs k-means clustering on a patch embedding matrix
###############################################################################

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


class CombinedSilhouetteInertiaClusterer:
    def __init__(
        self,
        k_min=1,
        k_max=8,
        k_avg_max=4,
        silhouette_threshold=0.15,
        relative_threshold=0.15,
        random_state=0,
        min_patch_size=5,
        n_smooth_iter=2,
        ignore_label=-1,
        plot=False,
        gifs_dir = "./"
    ):
        """
        Parameters
        ----------
        k_min : int
            Minimum number of clusters for KMeans.
        k_max : int
            Maximum number of clusters for KMeans.
        k_avg_max : int
            Upper bound on k after combining elbow & silhouette if they disagree.
        silhouette_threshold : float
            Minimum silhouette score at k=2 to justify splitting.
        relative_threshold : float
            Minimum % improvement in inertia from k=1→k=2 to justify splitting.
        random_state : int
            RNG seed for KMeans.
        min_patch_size : int
            Patches smaller than this threshold are smoothed.
        n_smooth_iter : int
            Number of smoothing iterations.
        ignore_label : int
            Label to ignore in smoothing step.
        """
        self.k_min = k_min
        self.k_max = k_max
        self.k_avg_max = k_avg_max
        self.silhouette_threshold = silhouette_threshold
        self.relative_threshold = relative_threshold
        self.random_state = random_state

        self.min_patch_size = min_patch_size
        self.n_smooth_iter = n_smooth_iter
        self.ignore_label = ignore_label
        self.plot = False #plot
        self.gifs_dir = gifs_dir

        self.final_k = None
        self.final_labels_1d = None    
        self.smoothed_labels_2d = None 
        self.kmeans_frame_files = []
    
    ##############################
    # Helper functions
    ##############################

    def combined_silhouette_inertia_clustering(
        self,
        X, 
        k_min=1, 
        k_max=8, 
        k_avg_max=4,
        silhouette_threshold=0.2, 
        relative_threshold=0.05, 
        random_state=0
    ):
        """
        Runs KMeans for k in [k_min..k_max] exactly once each,
        collects silhouette scores & inertias, and returns best_k.
        """
        n_samples = len(X)
        if n_samples < 2:
            return 1, np.zeros(n_samples, dtype=int), [None], [None]

        # --- Fit once for k=1 ---
        km1 = KMeans(n_clusters=1, random_state=random_state).fit(X)
        inertia_k1 = km1.inertia_ / n_samples
        silhouette_k1 = None  # undefined for k=1

        # If k_max=1, no reason to check further
        if k_max < 2:
            return 1, km1.labels_, [silhouette_k1], [inertia_k1]

        # --- Fit once for k=2 ---
        km2 = KMeans(n_clusters=2, random_state=random_state).fit(X)
        inertia_k2 = km2.inertia_ / n_samples
        sil_k2 = silhouette_score(X, km2.labels_)
        relative_improvement = (inertia_k1 - inertia_k2) / inertia_k1

        # If improvement is too small or silhouette is too low => remain at k=1
        if (relative_improvement < relative_threshold) or (sil_k2 < silhouette_threshold):
            return 1, km1.labels_, [silhouette_k1, sil_k2], [inertia_k1, inertia_k2]

        # --- Otherwise fit k=2..k_max and gather inertias & silhouettes ---
        all_k = range(2, k_max + 1)
        kmeans_models = {}
        inertias = []
        silhouettes = []

        # We already have k=2
        kmeans_models[2] = km2
        inertias.append(inertia_k2)
        silhouettes.append(sil_k2)

        for k in range(3, k_max + 1):
            km = KMeans(n_clusters=k, random_state=random_state).fit(X)
            kmeans_models[k] = km
            
            norm_inertia = km.inertia_ / n_samples
            inertias.append(norm_inertia)
            
            # If k>n_samples, silhouette_score is meaningless, but in normal usage k<<n_samples
            sil_val = silhouette_score(X, km.labels_) if k <= n_samples else -1
            silhouettes.append(sil_val)
            
        # (a) Silhouette-based best_k_sil
        best_idx_sil = np.argmax(silhouettes)
        best_k_sil = best_idx_sil + 2  

        # (b) Inertia-based best_k_elbow
        k_candidates = np.arange(2, k_max + 1)
        if len(k_candidates) == 1:
            best_k_elbow = 2
        else:
            kn = KneeLocator(k_candidates, inertias, curve="convex", direction="decreasing")
            best_k_elbow = kn.elbow
            if best_k_elbow is None:
                print("No elbow found => default to k=1.")
                best_k_elbow = 1  # fallback

        print(f"Silhouette-based best_k={best_k_sil}, elbow-based best_k={best_k_elbow}")

        # Combine if there's disagreement
        if best_k_sil == best_k_elbow:
            final_k = max(1, min(best_k_sil, k_avg_max)) # best_k_sil
        else:
            avg_k = 0.5 * (best_k_sil + best_k_elbow)
            final_k = int(math.ceil(avg_k))
            final_k = max(1, min(final_k, k_avg_max))

        assert (final_k <= k_avg_max), f"Final k={final_k} is greater than k_avg_max={k_avg_max}"

        # Get final labels from the chosen KMeans model
        if final_k == 1:
            final_labels = km1.labels_
        else:
            final_labels = kmeans_models[final_k].labels_

        return final_k, final_labels, [silhouette_k1] + silhouettes, [inertia_k1] + inertias


    def compute_region_statistics(self, label_map, heatmap, visited_indices, episode_num=0, step_num=0):
        """
        Computes region statistics for the current smoothed label map.
        """
        # Flatten the cluster map and the heatmap to handle indexing uniformly
        label_map_2d = label_map
        label_map_1d = label_map.ravel()
        heatmap_1d   = heatmap.ravel()

        # Identify unique labels (excluding ignore_label if present)
        unique_labels = np.unique(label_map_1d)
        region_dict = {}
        for lbl in unique_labels:
            if lbl == self.ignore_label:  
                continue
            region_dict[lbl] = {
                'num_patches': 0,
                'patches_visited': 0,
                'expectation': 0.0
            }

        # Accumulate totals for all patches
        total_patches = len(label_map_1d)
        for i in range(total_patches):
            lbl = label_map_1d[i]
            if lbl == self.ignore_label:
                continue
            region_dict[lbl]['num_patches'] += 1
            region_dict[lbl]['expectation'] += float(heatmap_1d[i])

        # # Exponential distribution (waiting time) = num_patches / expected_num_tgts
        for lbl in region_dict:
            region_dict[lbl]['expectation'] = region_dict[lbl]['num_patches'] / region_dict[lbl]['expectation']

        # Count only unique visited patches by converting to a set.
        unique_visited = set(visited_indices)
        for vi in unique_visited:
            if vi < 0 or vi >= total_patches:
                continue  
            lbl = label_map_1d[vi]
            if lbl == self.ignore_label:
                continue
            region_dict[lbl]['patches_visited'] += 1

        if self.plot:
            self.plot_cluster_map(label_map_2d, heatmap, visited_indices, region_dict, episode_num, step_num)

        return region_dict


    def plot_cluster_map(self, cluster_map, heatmap, path_taken, region_stats_dict, episode_num, step_num, cmap='tab20'):

        # 4) Plot (side-by-side) if requested
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        axes[0].imshow(cluster_map, cmap='tab20')
        axes[0].set_title(f"Raw KMeans Clusters")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap="viridis")
        axes[1].set_title("Heatmap")
        axes[1].axis('off')

        axes[2].imshow(cluster_map, cmap='tab20')
        axes[2].set_title("Raw KMeans Clusters")
        axes[2].axis('off')

        path_rows, path_cols = [], []
        for i, idx in enumerate(path_taken):
            rr = idx // cluster_map.shape[1]
            cc = idx % cluster_map.shape[1]
            path_rows.append(rr)
            path_cols.append(cc)
        axes[2].plot(path_cols, path_rows, c="r", linewidth=2)
        axes[2].plot(path_cols[-1], path_rows[-1], markersize=12, zorder=99, marker="^", ls="-", c="r", mec="black")
        axes[2].plot(path_cols[0], path_rows[0], 'co', c="r", markersize=8, zorder=5)

        # Create legend patches for each region.
        unique_labels = sorted(region_stats_dict.keys())
        max_label = max(unique_labels) if unique_labels else 1
        cm = plt.get_cmap(cmap)
        legend_patches = []
        for lbl in unique_labels:
            # Normalize the label to [0,1] for colormap lookup.
            norm_value = lbl / max_label if max_label > 0 else 0.5
            color = cm(norm_value)
            patch = mpatches.Patch(color=color, label=f"R{lbl}")
            legend_patches.append(patch)

        # Add legends to both subplots.
        axes[0].legend(handles=legend_patches, title="Regions", loc='upper right')
        axes[2].legend(handles=legend_patches, title="Regions", loc='upper right')

        # Build the legend text for each region using the provided format:
        legend_lines = []
        for label, stats in region_stats_dict.items():
            line = f"R{label}: patches={stats['num_patches']}, E={stats['expectation']:.3f}, visited={stats['patches_visited']}"
            legend_lines.append(line)
        legend_text = "\n".join(legend_lines)

        # Add the legend text as a subtitle at the bottom of the figure
        fig.text(0.5, 0.05, legend_text, ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Adjust layout to reserve space for the legend text
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  

        if not os.path.exists(self.gifs_dir):
            os.makedirs(self.gifs_dir)

        plt.savefig(f'{self.gifs_dir}/kmeans_{episode_num}_{step_num}.png'.format(dpi=150))
        self.kmeans_frame_files.append(f'{self.gifs_dir}/kmeans_{episode_num}_{step_num}.png')
        plt.close()


    def get_label_id(self, label_map, patch_idx):
        return label_map.ravel()[patch_idx]


    def get_probs(self, patch_idx, heatmap):
        return heatmap.ravel()[patch_idx]


    ##############################
    # Main functions
    ##############################
    def fit_predict(self, patch_embeds, map_shape):
        """
        Main function to obtain smoothed labelmap
        """
        # 1) Run combined silhouette & inertia
        best_k, final_labels, silhouettes, inertias = self.combined_silhouette_inertia_clustering(
            X=patch_embeds,
            k_min=self.k_min,
            k_max=self.k_max,
            k_avg_max=self.k_avg_max,
            silhouette_threshold=self.silhouette_threshold,
            relative_threshold=self.relative_threshold,
            random_state=self.random_state
        )
        self.final_k = best_k
        self.final_labels_1d = final_labels.copy()  

        # 2) Reshape for display
        H, W = map_shape
        cluster_map = final_labels.reshape(H, W)

        # 3) Apply smoothing
        cluster_map_smoothed = cluster_map
        self.smoothed_labels_2d = cluster_map_smoothed.copy()

        # 5) Return the smoothed 2D labels
        return self.smoothed_labels_2d
