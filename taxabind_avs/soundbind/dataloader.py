##############################################################################
# Name: dataloader.py
#
# - Handles loading of quadmodal dataset
# - https://huggingface.co/datasets/derektan95/avs-bench
###############################################################################

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config_sound import config
from sound_encoder import get_audio_clap
from datasets import load_dataset


class INatDataset(Dataset):
    def __init__(self,
                 data_file,
                 mode='train'): 
        
        # Load from huggingface dataset
        ds_config = data_file.split("|")[0]
        ds_split = data_file.split("|")[1]
        self.data_file = load_dataset(config.avs_dataset, name=ds_config, split=ds_split).to_pandas() 
        print("Loaded huggingface dataset: ", ds_config, ds_split)

        self.mode = mode
        if mode=='train':
            self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.GaussianBlur(5, (0.01, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        self.species_text = self.data_file['scientific_name'].tolist()
        self.species_classes = list(set(self.species_text))
        print("mode: ", self.mode)
        print("len(self.data_file): ", len(self.data_file))

    def __len__(self):
        return len(self.data_file)
        
    def get_sample(self,idx):
        sample = self.data_file.iloc[idx]
        id = sample.id
        sound_format = sample.sound_format
        data_path = config['data_path_train'] if self.mode == 'train' else config['data_path_val']
        image_path = os.path.join(data_path,"images",str(id)+".jpg")
        sound_path = os.path.join(data_path,"sounds_mp3",str(id)+"."+'mp3')
        sound = get_audio_clap(sound_path) 
        
        for k in sound.keys():
            sound[k] = sound[k].squeeze(0)
        image = self.transform(Image.open(image_path).convert("RGB"))

        return image, sound

    def __getitem__(self, idx):
        image, sound = self.get_sample(idx)
        return image, sound, self.species_classes.index(self.data_file.iloc[idx]['scientific_name'])
