##############################################################################
# Name: clip_vision_per_patch_model.py
#
# - Overloads CLIP template with custom functions
###############################################################################

import torch
from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig

class CLIPVisionPerPatchModel(CLIPVisionModelWithProjection):
    """
    Like CLIPVisionModelWithProjection but returns
    per-patch embeddings instead of pooled CLS tokens.
    """
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        # everything else (self.vision_model, self.visual_projection)
        # is set up for you by the parent class

    def forward(self, pixel_values, **kwargs):
        # 1) run the ViT backbone → last_hidden_state [B, n_patches, hidden_size]
        outputs = self.vision_model(pixel_values, return_dict=True, **kwargs)
        hidden_states = outputs.last_hidden_state

        # 2) project every patch token → [B, n_patches, projection_dim]
        patch_embeds = self.visual_projection(hidden_states)

        # 3) Postprocessing embeds
        patch_embeds = torch.nn.functional.normalize(patch_embeds, dim=-1)
        patch_embeds = patch_embeds.squeeze()   # (Patches, proj_dim)

        return patch_embeds