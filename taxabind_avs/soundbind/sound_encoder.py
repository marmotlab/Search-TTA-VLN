##############################################################################
# Name: sound_encoder.py
#
# - Wrapper for sound CLAP model
###############################################################################

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
import torch
import torchaudio
from transformers import ClapProcessor
from transformers import ClapAudioModelWithProjection
from config_sound import config


processor = ClapProcessor.from_pretrained(config.sound_encoder)
SAMPLE_RATE = 48000

def get_audio_clap(path_to_audio,format="mp3",padding="repeatpad",truncation="fusion"):
    track, sr = torchaudio.load(path_to_audio, format=format)  
    track = track.mean(axis=0)
    track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=SAMPLE_RATE)
    output = processor(audios=track, sampling_rate=SAMPLE_RATE, max_length_s=10, return_tensors="pt",padding=padding,truncation=truncation)
    return output


class CLAP_audiomodel_withProjection(pl.LightningModule):
    def __init__(self,freeze=False):
        super().__init__()
        if freeze:
            self.model = ClapAudioModelWithProjection.from_pretrained(config.sound_encoder).eval()
            for params in self.model.parameters():
                params.requires_grad=False
        else:
            self.model = ClapAudioModelWithProjection.from_pretrained(config.sound_encoder).train()
    def forward(self,audio):
        batch_embeddings_audio = self.model(**audio)['audio_embeds']
        return batch_embeddings_audio
    
if __name__ == '__main__':
    path_to_audio ="/mnt/hdd/inat2021_ds/sound_train/sounds_mp3/165878447.mp3"
    sample =  get_audio_clap(path_to_audio)
    print(sample.keys())
    
    sample['input_features'] = torch.concat([sample['input_features'],sample['input_features']],axis=0)
    sample['is_longer'] = torch.concat([sample['is_longer'],sample['is_longer']],axis=0)
    print(sample['input_features'].shape,sample['is_longer'].shape) 
    model = CLAP_audiomodel_withProjection(freeze=False)
    audio_feat = model(sample)
    print(audio_feat.shape) 