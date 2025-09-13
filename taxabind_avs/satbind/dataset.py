##############################################################################
# Name: dataset.py
#
# - Handles loading of trimodal dataset
# - https://huggingface.co/datasets/derektan95/avs-bench
###############################################################################

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset 
from config_sat import config


class SatNatDataset(Dataset):
    def __init__(self, img_dir, imo_dir, json_path, sat_to_img_ids_path, patch_size, mode='train', get_img_path=False, sat_to_img_ids_json_is_train_dict=True, tax_to_filter_val="", sound_dir=None):
        self.img_dir = img_dir
        self.imo_dir = imo_dir
        self.patch_size = patch_size
        self.get_img_path = get_img_path
        self.mode = mode
        self.sat_to_img_ids_json_is_train_dict = sat_to_img_ids_json_is_train_dict
        self.tax_to_filter_val = tax_to_filter_val
        self.sound_dir = sound_dir
        self.current_epoch = 0

        self.json = json.load(open(json_path, 'r'))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = [d for d in self.images if d['latitude'] is not None and d['longitude'] is not None]
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        self.inat_json_dict = {
            "images": {img["id"]: img for img in self.images},
            "annotations": {ann["id"]: ann for ann in self.annot},
        }

        # Load from huggingface dataset
        ds_config = sat_to_img_ids_path.split("|")[0]
        ds_split = sat_to_img_ids_path.split("|")[1]
        self.sat_to_img_ids_json = load_dataset(config.avs_dataset, name=ds_config, split=ds_split)   
        print("Loaded huggingface dataset: ", ds_config, ds_split)

        # Expand dict
        self.sat_to_img_ids_tuples = []
        if self.sat_to_img_ids_json_is_train_dict:
            # Convert from a huggingface list of dicts into dict of dicts (no duplicate keys)
            self.sat_to_img_ids_json = {sat_sample["sat_key"]: sat_sample for sat_sample in self.sat_to_img_ids_json}
            for sat_key, sat_sample in self.sat_to_img_ids_json.items():
                id = sat_sample["id"] 
                sat_path = sat_sample["sat_path"]
                img_ids = sat_sample["img_ids"]
                for img_id in img_ids:
                    self.sat_to_img_ids_tuples.append((id, sat_path, img_id))
            print("len(self.sat_to_img_ids_json): ", len(self.sat_to_img_ids_json))
            print("len(self.sat_to_img_ids_tuples): ", len(self.sat_to_img_ids_tuples))
        else:
            self.filtered_val_ds_by_tax = [d for d in self.sat_to_img_ids_json if self.tax_to_filter_val in d['taxonomy']] 

        if mode == 'train':
            self.img_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.GaussianBlur(5, (0.01, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            self.imo_transform = transforms.Compose([
                transforms.Resize((336,336)),
                transforms.GaussianBlur(5, (0.01, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            self.imo_transform = transforms.Compose([
                transforms.Resize((336,336)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        self.debug_img_viz_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224))        
        ])
        self.debug_imo_viz_transform = transforms.Compose([
            transforms.Resize((336,336))        
        ])


    def __len__(self):
        return len(self.sat_to_img_ids_tuples)
        

    def __getitem__(self, idx):

        if not self.sat_to_img_ids_json_is_train_dict:
            print("Json is not dict. Please reformat for training!")
            exit()

        ## Pixel-level CLIP
        id, sat_path, img_id = self.sat_to_img_ids_tuples[idx]
        imo_path = os.path.join(self.imo_dir, sat_path)
        imo = self.imo_transform(Image.open(imo_path))
        sat_id = Path(sat_path).stem      

        img_path = os.path.join(self.img_dir, self.inat_json_dict["images"][img_id]["file_name"])
        img = self.img_transform(Image.open(img_path))

        # # Map lat-lon to pixel in sat img
        sat_min_lon = self.sat_to_img_ids_json[sat_id]["sat_bounds"]["min_lon"]
        sat_min_lat = self.sat_to_img_ids_json[sat_id]["sat_bounds"]["min_lat"]
        sat_max_lon = self.sat_to_img_ids_json[sat_id]["sat_bounds"]["max_lon"]
        sat_max_lat = self.sat_to_img_ids_json[sat_id]["sat_bounds"]["max_lat"]
        
        img_lon = self.inat_json_dict["images"][img_id]["longitude"]
        img_lat = self.inat_json_dict["images"][img_id]["latitude"]
        row, col = self.latlon_to_pixel(img_lat, img_lon, sat_min_lat, sat_max_lat, sat_min_lon, sat_max_lon, imo.shape[2], imo.shape[1])

        patch_idx = self.pixel_to_patch_idx(row, col, self.patch_size, imo.shape[2], imo.shape[1])
        patch_idx += 1  # account for [CLS] token at the start of ViT input sequence

        species_text = " ".join(self.inat_json_dict["images"][img_id]['file_name'].split("/")[1].split("_")[1:])

        if self.get_img_path:
            return img_path, imo_path, img, imo, self.inat_json_dict["annotations"][img_id]['category_id'], patch_idx, species_text, self.species_text.index(species_text)
        else:
            return img, imo, self.inat_json_dict["annotations"][img_id]['category_id'], patch_idx, species_text, self.species_text.index(species_text)


    def latlon_to_pixel(self, lat, lon, lat_min, lat_max, lon_min, lon_max, img_width, img_height):
        lat_res = (lat_max - lat_min) / img_height
        lon_res = (lon_max - lon_min) / img_width
        col = int(math.floor((lon - lon_min) / lon_res))
        row = int(math.floor((lat_max - lat) / lat_res))
        return row, col
    

    def pixel_to_patch_idx(self, row, col, patch_size, img_width, img_height):
        patch_size_width = patch_size
        patch_size_height = patch_size
        patch_row = row // patch_size_height
        patch_col = col // patch_size_width
        patch_idx = patch_row * (img_width // patch_size) + patch_col
        return patch_idx
    

    def set_epoch(self, epoch):
        self.current_epoch = epoch


    def get_search_ds_data(self, idx):

        if self.sat_to_img_ids_json_is_train_dict:
            print("Json is dict. Please reformat for target search!")
            exit()
        
        bounded_idx = idx % len(self.filtered_val_ds_by_tax)
        sat_sample = self.filtered_val_ds_by_tax[bounded_idx]
        target_positions = sat_sample["target_positions"]
        imo_path = os.path.join(self.imo_dir, sat_sample["sat_path"])
        imo = self.imo_transform(Image.open(imo_path))

        img_paths = []
        imgs = []
        species_texts = []
        for img_id in sat_sample["img_ids"]:
            img_path = os.path.join(self.img_dir, self.inat_json_dict["images"][img_id]["file_name"])
            img = self.img_transform(Image.open(img_path))
            img_paths.append(img_path)
            imgs.append(img)
            species_text = " ".join(self.inat_json_dict["images"][img_id]['file_name'].split("/")[1].split("_")[1:])
            species_texts.append(species_text)
        imgs = torch.stack(imgs)    

        if len(set(species_texts)) > 1:
            print("Species mismatch in search dataset!")
            exit()
        else:
            species_name = species_texts[0]
            gt_mask_name = str(sat_sample["id"]) + "_" + sat_sample["taxonomy"] + ".png"   
            gt_mask_name = gt_mask_name.replace(" ", "_")

        # Consider sound if valid 
        sounds, sound_ids = [], []
        if self.sound_dir is not None and "sound_ids" in sat_sample:
            sound_id = sat_sample["sound_ids"][0]
            sound_path = os.path.join(self.sound_dir,"sounds_mp3",str(sound_id)+"."+'mp3')

            from soundbind.sound_encoder import get_audio_clap
            sound = get_audio_clap(sound_path) 
            sounds.append(sound) 
            sound_ids.append(sound_id) 

        return img_paths, imo_path, imgs, imo, sounds, sound_ids, species_name, target_positions, gt_mask_name