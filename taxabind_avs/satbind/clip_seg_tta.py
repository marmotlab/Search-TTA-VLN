##############################################################################
# Name: clip_seg_tta.py
#
# - Performs TTA on sat encoder a collected measurements
###############################################################################

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
import torch
import numpy as np
import open_clip
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from dataset import SatNatDataset
from model_sat import SatBind
from transformers import ClapAudioModelWithProjection
from clip_vision_per_patch_model import CLIPVisionPerPatchModel
from types import SimpleNamespace
from config_sat import config
 

class ClipSegTTA:
    def __init__(
        self,
        img_dir: str,
        imo_dir: str,
        json_path: str,
        sat_to_img_ids_path: str,
        sat_checkpoint_path: str,
        load_pretrained_hf_ckpt: bool = True,
        sample_index: int = 0,  # Set using 'reset' 
        blur_kernel = (5,5),    # (0,0) for no gaussian blur
        batch_size: int = 1,
        num_workers: int = 1,
        device: str = "cuda",
        sat_to_img_ids_json_is_train_dict: bool = True,
        tax_to_filter_val: str = "",
        load_model: bool = True,
        query_modality: str = "image",    # image, text, sound
        sound_dir: str = None,
        sound_checkpoint_path: str = None,
    ):

        self.img_dir = img_dir
        self.imo_dir = imo_dir
        self.json_path = json_path
        self.sat_to_img_ids_path = sat_to_img_ids_path
        self.sat_checkpoint_path = sat_checkpoint_path
        self.pretrained_hf_ckpt = load_pretrained_hf_ckpt
        self.sample_index = sample_index    
        self.blur_kernel = blur_kernel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.sat_to_img_ids_json_is_train_dict = sat_to_img_ids_json_is_train_dict
        self.tax_to_filter_val = tax_to_filter_val
        self.load_model = load_model
        self.query_modality = query_modality
        self.sound_dir = sound_dir
        self.sound_checkpoint_path = sound_checkpoint_path

        # Prepare the dataset
        start_time = time.time()
        self.load_data()
        print(f"Dataset loaded in {(time.time()-start_time):.2f}s.")

        if self.load_model:
            start_time = time.time()

            # Load the global model (original/frozen checkpoint)
            self.load_global_model()
            self.tokenizer = open_clip.get_tokenizer(config.image_encoder_finetuned)
            print(f"Global model loaded in {(time.time()-start_time):.2f}s.")

            # Create the local model that will be adapted for TTA
            if self.pretrained_hf_ckpt:
                imo_encoder = CLIPVisionPerPatchModel.from_pretrained(config.sat_encoder_finetuned)
                imo_encoder.to(self.device)
                imo_encoder.eval()
                bio_model, *_ = open_clip.create_model_and_transforms(config.image_encoder_finetuned)
                bio_model.to(self.device)
                bio_model.eval()
                logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                self.model_local = SimpleNamespace(imo_encoder=imo_encoder, bio_model=bio_model, logit_scale=logit_scale)

                # Load sound model if provided
                if self.query_modality =="sound":
                    self.sound_model = ClapAudioModelWithProjection.from_pretrained(config.sound_encoder_finetuned)
                    self.sound_model.to(self.device)
                    self.sound_model.eval()
                print("~Loaded HF checkpoint")
            else:
                self.model_local = SatBind(train_dataset=None, val_dataset=None)
                self.model_local.to(self.device)
                self.model_local.eval()

                # Load sound model if provided
                if self.query_modality =="sound" and self.sound_checkpoint_path:
                    from soundbind.model_sound import AudioBind
                    self.sound_model = AudioBind.load_from_checkpoint(self.sound_checkpoint_path, train_dataset=None, val_dataset=None)
                    self.sound_model.to(self.device)
                    self.sound_model.eval()
                print("~Loaded local checkpoint")

        self.reset(sample_idx=self.sample_index)
        self.clip_inference_time = 0.0
        self.tta_time = 0.0


    def load_data(self):
        """Load or initialize the dataset."""
        self.dataset = SatNatDataset(
            img_dir=self.img_dir,
            imo_dir=self.imo_dir,
            json_path=self.json_path,
            sat_to_img_ids_path=self.sat_to_img_ids_path,
            patch_size=config.patch_size,
            mode="val",
            get_img_path=True,
            sat_to_img_ids_json_is_train_dict=self.sat_to_img_ids_json_is_train_dict,
            tax_to_filter_val=self.tax_to_filter_val,
            sound_dir=self.sound_dir
        )

    def reset(self, sample_idx):
        """Reset the parameters & local model for the current sample."""
        if self.load_model:
            self.reset_local_model()    # Reset to global weights as init

        self.img_paths, self.imo_path, self.imgs, self.imo, self.sounds, self.sound_ids, self.species_name, self.target_positions, self.gt_mask_name = self.dataset.get_search_ds_data(sample_idx)
        self.imgs = self.imgs.to(self.device)
        self.tgts_gt_score = None
        if self.load_model:
            self.heatmap, self.heatmap_unnormalized, self.heatmap_unnormalized_initial, self.patch_embeds = None, None, None, None
            img = self.imgs[0].unsqueeze(0).to(self.device)
            imo = self.imo.unsqueeze(0).to(self.device)  
            txt = [self.species_name]
            if self.sounds != []:
                sound = self.sounds[0].to(self.device)
                for k in sound.keys():
                    sound[k] = sound[k].to(self.device)
            else:
                sound = None
            self.generate_heatmap(img, imo, txt, sound=sound, modality=self.query_modality)   

            # Find avg heatmap score for target positions 
            scores = []
            imo_orig = Image.open(self.imo_path)
            for pos in self.target_positions:
                row_trans = int(pos[0] * self.heatmap.shape[0] / imo_orig.size[0])
                col_trans = int(pos[1] * self.heatmap.shape[1] / imo_orig.size[1])
                scores.append(self.heatmap[row_trans, col_trans])
            self.tgts_gt_score = np.mean(scores)
            

    def load_global_model(self):
        """Load the global SatBind model from checkpoint, move to device, and eval."""
        if self.pretrained_hf_ckpt:
            print("Downloading HF checkpoint (if not already downloaded)...")
            self.model_global = CLIPVisionPerPatchModel.from_pretrained(config.sat_encoder_finetuned)
        else:
            self.model_global = SatBind.load_from_checkpoint(
                self.sat_checkpoint_path, train_dataset=None, val_dataset=None
            )
        self.model_global = self.model_global.to(self.device)
        self.model_global.eval()


    def reset_local_model(self):
        """
        Reset the local model to match the global model's parameters
        and freeze/unfreeze layers for TTA.
        """
        start_time = time.time()
        with torch.no_grad():
            local_params = self.model_local.imo_encoder.parameters() \
                if self.pretrained_hf_ckpt else self.model_local.parameters()
            for param_global, param_local in zip(
                self.model_global.parameters(), local_params
            ):
                param_local.data.copy_(param_global.data)

        if self.pretrained_hf_ckpt:
            for param in self.model_local.imo_encoder.parameters():
                param.requires_grad = True
            self.model_local.imo_encoder.eval()
        else:
            # Freeze everything except the satellite encoder & custom projection
            for name, param in self.model_local.named_parameters():
                if "imo_encoder" in name or "visual_projection_custom" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.model_local.eval()


    def execute_tta(
        self,
        patch_indices: list,
        patch_is_pos: list,
        pos_sample_weight: float,
        neg_sample_weight: float,        
        tta_steps: int = 10,
        lr: float = 2e-6,
        reset_weights: bool = True,
        num_viz_steps: int = 1,
        viz_heatmap: bool = False,
    ):
        """
        Run test-time adaptation using the local model. The local model is first
        reset to the global weights. After TTA, the global model remains
        unchanged; only the local model is updated.
        """

        ### Option 1: SAMPLE FROM DATASET
        # 1) Reset the local model to global weights
        if reset_weights:
            self.reset_local_model()

        # 2) Prepare the sample(s) for TTA
        img = self.imgs[0].unsqueeze(0).to(self.device)
        imo = self.imo.unsqueeze(0).to(self.device)         # vectorize 
        txt = [self.species_name]
        if self.sounds != []:
            sound = self.sounds[0].to(self.device)
            for k in sound.keys():
                sound[k] = sound[k].to(self.device)
        else:
            sound = None
        patch_indices = [idx+1 for idx in patch_indices]    # Consider the [CLS] token offset
        patch_idx = torch.tensor(patch_indices).to(self.device)
        
        # ---------------------------------------------------------------------

        # 5) Set up optimizer 
        local_params = self.model_local.imo_encoder.parameters() \
            if self.pretrained_hf_ckpt else self.model_local.parameters()
        optimizer = torch.optim.Adam(
            [p for p in local_params if p.requires_grad], lr=lr
        )
        start_time = time.time()

        # 6) TTA loop
        for step in range(tta_steps):
            batch_size = imo.shape[0]

            # Query embeds
            query_embeds = self.generate_query_embeds(img, imo, txt, sound=sound, modality=self.query_modality)

            # Sat Embeds
            if self.pretrained_hf_ckpt:
                imo_embeds = self.model_local.imo_encoder.vision_model(imo, return_dict=True).last_hidden_state
                imo_embeds = imo_embeds[torch.arange(batch_size), patch_idx] 
                imo_embeds = self.model_local.imo_encoder.visual_projection(imo_embeds)
            else:
                imo_embeds = self.model_local.imo_encoder(imo).last_hidden_state    # (batch, Patches, hidden_dim)
                imo_embeds = imo_embeds[torch.arange(batch_size), patch_idx]        # (batch, hidden_dim)
                imo_embeds = self.model_local.visual_projection_custom(imo_embeds)  # (batch_size, proj_dim)
            imo_embeds = torch.nn.functional.normalize(imo_embeds, dim=-1)

            # Compute Similarity Loss
            logit_scale = self.model_local.logit_scale.exp()
            similarity = imo_embeds @ query_embeds.t() * logit_scale

            # Negative Log Likelihood loss for spatial poisson point process
            patch_probs = similarity.squeeze().sigmoid()
            counts = torch.tensor(patch_is_pos, dtype=torch.float32, device=similarity.device)
            pos_weights = torch.tensor(pos_sample_weight, dtype=torch.float32, device=similarity.device)
            neg_weights = torch.tensor(neg_sample_weight, dtype=torch.float32, device=similarity.device)
            loss = (neg_weights * patch_probs - pos_weights * counts * torch.log(patch_probs + 1e-6))
            loss = loss.sum()

            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.tta_time = time.time() - start_time

            # Visualization every 'num_viz_steps' steps (if enabled)
            if (step + 1) % num_viz_steps == 0 and viz_heatmap:
                self.generate_heatmap(img, imo, txt, sound=sound, modality=self.query_modality)
                self.visualize_heatmap(
                    step=step,
                    img_path_viz=self.img_paths[0], # Viz 1st image
                    imo_path_viz=self.imo_path,
                    patch_idx_viz=patch_idx,
                    patch_is_pos=patch_is_pos,
                    species_name=self.species_name
                )

        # Save final heatmap after TTA steps 
        self.generate_heatmap(img, imo, txt, sound=sound, modality=self.query_modality)


    def generate_query_embeds(self, img, imo, txt, sound=None, modality="image"):

        # Query Embeds
        if modality == "image":
            query_embeds, *_ = self.model_local.bio_model(img)        # (batch_size, proj_dim)
            if query_embeds.shape[0] > 1:
                query_embeds = query_embeds.mean(dim=0, keepdim=True) # (1, proj_dim)
        elif modality == "text":
            txt_tokenized = self.tokenizer(txt).to(imo.device)
            _, query_embeds, _ = self.model_local.bio_model(text=txt_tokenized)
        elif modality == "sound":
            if sound == None: 
                print("!!!! Sound modality requires sound input !!!")
                exit(1)
            if self.pretrained_hf_ckpt:
                unnormalized_audio_embeds = self.sound_model(**sound).audio_embeds
            else:
                unnormalized_audio_embeds = self.sound_model.audio_encoder(sound)
            query_embeds = torch.nn.functional.normalize(unnormalized_audio_embeds, dim=-1)
        else:
            raise ValueError("Invalid modality")
        
        return query_embeds


    def generate_heatmap(self, img, imo, txt, sound=None, modality="image"):

        start_time = time.time()

        # Satellite encoder outputs
        if self.pretrained_hf_ckpt:
            imo_embeds = self.model_local.imo_encoder(imo)
        else:
            imo_embeds = self.model_local.imo_encoder(imo).last_hidden_state
            imo_embeds = self.model_local.visual_projection_custom(imo_embeds)
            imo_embeds = torch.nn.functional.normalize(imo_embeds, dim=-1)

        # Remove batch dimension -> (num_tokens, proj_dim)
        imo_embeds = imo_embeds.squeeze(0)
        self.patch_embeds = imo_embeds.clone()[1:].cpu().detach().numpy()

        # Ground image embedding (bio CLIP model)
        query_embeds = self.generate_query_embeds(img, imo, txt, sound=sound, modality=modality)

        # Same logit scale as in SatBind
        logit_scale = self.model_local.logit_scale.exp()
        sim = query_embeds @ imo_embeds.t() * logit_scale
        # Sigmoid to get similarity scores
        scores = sim.t().sigmoid()  # (num_tokens, 1)

        # Exclude [CLS] token at index 0
        score_no_cls = scores[1:].squeeze()  # shape: (num_tokens-1,)
        num_tokens = score_no_cls.shape[0]
        side_dim = int(num_tokens**0.5)
        sim_scores = score_no_cls.reshape(side_dim, side_dim).clone()
        sim_scores = sim_scores.cpu().detach().numpy()

        self.clip_inference_time = time.time() - start_time

        # Gausian Smoothing
        if self.blur_kernel != (0,0):
            sim_scores = cv2.GaussianBlur(sim_scores, self.blur_kernel, 0)

        # Normalize to expectation
        self.heatmap_unnormalized = sim_scores
        scale = len(self.target_positions) / (self.heatmap_unnormalized.sum())
        self.heatmap_unnormalized *= scale
        if self.heatmap_unnormalized_initial is None:
            self.heatmap_unnormalized_initial = self.heatmap_unnormalized.copy()

        # Standard normalization to (0,1)
        self.heatmap = sim_scores.copy()
        self.heatmap = (self.heatmap - self.heatmap.min()) / (self.heatmap.max() - self.heatmap.min())  


    def visualize_heatmap(
        self,
        step: int,
        img_path_viz: str,
        imo_path_viz: str,
        patch_idx_viz: torch.Tensor,
        patch_is_pos: list,
        species_name: str
    ):
        """
        Visualization function that plots the ground image, satellite image with
        highlighted patch, and the learned heatmap.
        """

        # Switch off gradients for visualization
        with torch.no_grad():
            side_dim = self.heatmap.shape[0]

            # -----------------------------------------------------------------
            # Highlight the patch in the satellite image
            sat_img_orig = Image.open(imo_path_viz)
            sat_highlight = np.array(
                self.dataset.debug_imo_viz_transform(sat_img_orig.copy())
            )

            for idx, patch_idx in enumerate(patch_idx_viz):

                # Because patch_idx includes the [CLS] offset, subtract 1
                patch_idx_actual = patch_idx - 1

                # Get dimensions (H x W)
                H, W = sat_highlight.shape[0], sat_highlight.shape[1]

                # Number of patches in each dimension
                patches_per_col = W // config.patch_size
                patches_per_row = H // config.patch_size

                # Determine row/col in the patch grid
                patch_row = patch_idx_actual // patches_per_col
                patch_col = patch_idx_actual % patches_per_row

                # Pixel boundaries
                x_start = patch_col * config.patch_size
                x_end = (patch_col + 1) * config.patch_size
                y_start = patch_row * config.patch_size
                y_end = (patch_row + 1) * config.patch_size

                # Blue color for positive patches (transparent)
                if patch_is_pos[idx]:
                    sat_highlight[y_start:y_end, x_start:x_end, 0] = 0
                    sat_highlight[y_start:y_end, x_start:x_end, 1] = 0
                    sat_highlight[y_start:y_end, x_start:x_end, 2] = 255
                # Red color for negative patches (transparent)
                else:
                    sat_highlight[y_start:y_end, x_start:x_end, 0] = 255
                    sat_highlight[y_start:y_end, x_start:x_end, 1] = 0
                    sat_highlight[y_start:y_end, x_start:x_end, 2] = 0


            # -----------------------------------------------------------------
            # Plot results
            fig, axes = plt.subplots(1, 3, figsize=(12, 6))
            fig.suptitle(f"Query: {species_name}")

            # Ground image
            img_orig = Image.open(img_path_viz)
            axes[0].imshow(img_orig)
            axes[0].set_title("Ground Image")
            axes[0].axis("off")

            # Satellite image
            axes[1].imshow(sat_highlight)
            axes[1].set_title("Sat Image")
            axes[1].axis("off")

            # Heatmap
            heatmap_np = self.heatmap_unnormalized
            im = axes[2].imshow(heatmap_np, cmap="viridis")
            axes[2].set_title(
                f"Heatmap at TTA Step {step:03d} ({side_dim}x{side_dim})"
            )
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":

    ###########################################
    # Example with Image Modality
    ###########################################
    
    clip_seg_tta = ClipSegTTA(
        img_dir="/mnt/hdd/avs_bench_ds/inat21",
        imo_dir="/mnt/hdd/avs_bench_ds/sat_jpg/train_512px",
        json_path="/mnt/hdd/avs_bench_ds/inat21/train.json",
        sat_to_img_ids_path="search_eval_trimodal|val_in_domain",
        patch_size=14,
        load_pretrained_hf_ckpt=True,
        sat_checkpoint_path="",
        sample_index=261,
        batch_size=1,
        num_workers=1,
        device="cuda",
        sat_to_img_ids_json_is_train_dict=False,
        query_modality="image"
    )

    # Image modality test
    patch_indices = [50, 357] 
    patch_is_pos = [True, False]
    pos_sample_weight = 1.0
    neg_sample_weight = 1.0
    clip_seg_tta.execute_tta(
        patch_indices, 
        patch_is_pos, 
        pos_sample_weight,
        neg_sample_weight,
        tta_steps=10, # for sanity check
        num_viz_steps=2, 
        viz_heatmap=True
    )

    ###########################################
    # Example with Sound Modality
    ###########################################

    # # Sound Modality Test
    # clip_seg_tta = ClipSegTTA(
    #     img_dir="/mnt/hdd/avs_bench_ds/inat21",
    #     imo_dir="/mnt/hdd/avs_bench_ds/sat_jpg/train_512px",
    #     json_path="/mnt/hdd/avs_bench_ds/inat21/train.json",
    #     sat_to_img_ids_path="search_eval_quadrimodal|val_in_domain",
    #     sound_dir='/mnt/hdd/avs_bench_ds/sound_mp3/test',
    #     patch_size=14,
    #     sat_checkpoint_path="", 
    #     sound_checkpoint_path = "",
    #     sample_index=120,   
    #     batch_size=1,
    #     num_workers=1,
    #     device="cuda",
    #     sat_to_img_ids_json_is_train_dict=False,
    #     query_modality="sound"
    # )

    # # Sound modality test
    # patch_indices = [422, 32] 
    # patch_is_pos = [True, False]
    # pos_sample_weight = 1.0
    # neg_sample_weight = 1.0
    # clip_seg_tta.execute_tta(
    #     patch_indices, 
    #     patch_is_pos, 
    #     pos_sample_weight,
    #     neg_sample_weight,
    #     tta_steps=30, 
    #     num_viz_steps=2, 
    #     viz_heatmap=True
    # )