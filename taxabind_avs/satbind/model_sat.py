##############################################################################
# Name: model_sat.py
#
# - Training wrapper for satellite image CLIP model
###############################################################################

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection
from dataset import SatNatDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from config_sat import config


def create_pairwise_mask(labels):
    labels = labels.reshape(-1)
    num_samples = len(labels)
    pairwise_mask = torch.zeros(num_samples, num_samples).to(labels.device)

    for i in range(num_samples):
        pairwise_mask[i, :] = (labels == labels[i])

    return pairwise_mask

def clip_loss(similarity: torch.Tensor, label) -> torch.Tensor:
    overhead_img_loss = contrastive_loss(similarity, label)
    ground_img_loss = contrastive_loss(similarity.t(), label.t())
    return 0.5*torch.mean(torch.sum(overhead_img_loss, dim=-1)) + 0.5*torch.mean(torch.sum(ground_img_loss, dim=-1))

def contrastive_loss(logits: torch.Tensor, label) -> torch.Tensor:
    gt = create_pairwise_mask(label)
    return -gt*torch.log(logits.softmax(-1)+1e-6)


class SatBind(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        #initialize bio CLIP with frozen weights
        self.bio_model, *_ = open_clip.create_model_and_transforms(config.image_encoder_finetuned)
        if config.locked_tuning:
            for param in self.bio_model.parameters():
                param.requires_grad = False
        
        #initialize CLIP with trainable weights
        self.imo_encoder = CLIPVisionModelWithProjection.from_pretrained(config.sat_encoder).train()
        for layer in self.imo_encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = kwargs.get('batch_size', config.batch_size)
        self.lr = kwargs.get('lr', config.lr)

        # Custom
        clip_cfg = self.imo_encoder.config
        self.visual_projection_custom = nn.Linear(clip_cfg.hidden_size, 512, bias=False) # clip_cfg.projection_dim)


    def forward(self, batch):
        img, imo, label, patch_idx, *_ = batch
        batch_size = img.shape[0]

        #compute bioclip embeddings
        img_embeds, *_ = self.bio_model(img)    # (batch_size, proj_dim)
        
        # Similarity computation
        imo_embeds = self.imo_encoder(imo).last_hidden_state          # (batch, Patches, hidden_dim)
        imo_embeds = imo_embeds[torch.arange(batch_size), patch_idx]  # (batch, hidden_dim)
        imo_embeds = self.visual_projection_custom(imo_embeds)        # (batch_size, proj_dim)

        return img_embeds, imo_embeds, label

    
    def shared_step(self, batch, return_sim_matrix=False):
        
        img_embeds, imo_embeds, label, *_ = self(batch)
        imo_embeds = torch.nn.functional.normalize(imo_embeds, dim=-1)

        #exponentiate the log of temperrature
        logit_scale = self.logit_scale.exp()

        #compute similarity 
        img_to_imo_sim = img_embeds @ imo_embeds.t() * logit_scale

        if return_sim_matrix:
            img_to_imo_sim_copy = img_to_imo_sim.clone().detach()
        
        loss = clip_loss(img_to_imo_sim, label) 

        if return_sim_matrix:
            return loss, img_to_imo_sim_copy  
        else:
            return loss


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('temperature', self.logit_scale.data, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=config.num_workers,
                          shuffle=True,    # True
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=config.num_workers,
                          shuffle=False,
                          persistent_workers=False)

    def configure_optimizers(self):
        params = self.parameters()
        self.optim = torch.optim.AdamW(params,
                                       lr=self.lr,
                                       betas=(0.9,0.98),
                                       eps=1e-6
                                    )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optim,
            T_0=20,
            eta_min=1e-6
        )
        return [self.optim], [self.scheduler]   
    

if __name__ == '__main__':
    img_dir = config.img_dir
    imo_dir = config.imo_dir
    imo_dir_val = config.imo_dir_val
    train_json_path = config.train_json_path
    val_json_path = config.val_json_path
    sat_to_img_ids_train_json_path = config.sat_to_img_ids_train_json_path
    sat_to_img_ids_val_json_path = config.sat_to_img_ids_val_json_path
    patch_size = config.patch_size
    
    #define dataset
    train_dataset = SatNatDataset(img_dir, imo_dir, train_json_path, sat_to_img_ids_train_json_path, patch_size)
    val_dataset = SatNatDataset(img_dir, imo_dir_val, val_json_path, sat_to_img_ids_val_json_path, patch_size, mode='val')

    #define model
    model = SatBind(train_dataset=train_dataset, val_dataset=val_dataset)
    torch.cuda.empty_cache()

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.save_dir,
        filename=config.filename,
        mode='min',
        save_top_k=1,    
        save_last=True   
    )
    checkpoint.CHECKPOINT_NAME_LAST = config.filename + "-LAST"  

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true', # supress pl issues with 'unused trainable params'
        devices=config.devices, 
        max_epochs=config.max_epochs,
        num_nodes=1,
        callbacks=[checkpoint],
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=1,
        val_check_interval=config.val_check_interval,
        )
    
    if config.resume_from_checkpoint:
        trainer.fit(model, ckpt_path=f"{config.save_dir}/{config.resume_checkpoint_name}.ckpt")
    else:
        trainer.fit(model)