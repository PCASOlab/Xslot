import warnings

import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as f
from torch.nn import init
import random
import timm

from sklearn.cluster import AgglomerativeClustering
class Temporal_Binder(nn.Module):
    def __init__(self, slot_dim, num_slots, num_frame, mask_ratio=0.20):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.F = num_frame
        self.mask_ratio = mask_ratio  # Percentage of frames to mask

        encoder_layer = nn.TransformerEncoderLayer(self.slot_dim, nhead=8, dim_feedforward=4*self.slot_dim, batch_first=True)
        self.slot_transformer = nn.TransformerEncoder(encoder_layer, 3)

        self.pos_embed_temporal = nn.Parameter(torch.Tensor(1, self.F, 1, self.slot_dim))
        init.normal_(self.pos_embed_temporal, mean=0., std=.02)

    def random_masking(self, slots):
        """ Apply random masking to the F dimension of slots """
        B, F, S, D_slot = slots.shape
        num_to_mask = int(self.mask_ratio * F)  # Number of frames to mask
        
        # Generate a random mask of shape (B, F) with `num_to_mask` masked frames (1) and others unmasked (0)
        mask = torch.ones((B, F), dtype=torch.float, device=slots.device)
        for i in range(B):
            mask_indices = torch.randperm(F)[:num_to_mask]  # Randomly select `num_to_mask` indices to mask
            mask_indices = mask_indices*0+4
            mask[i, mask_indices] = 0  # Mark the selected frames as unmasked (0)

        # Apply the mask to slots: mask out (set to 0) the masked frames in the F dimension
        masked_slots = slots.clone()  # Clone the original slots
        mask_frame_level = mask
        mask = mask.unsqueeze(2).unsqueeze(3)  # Shape becomes [4, 5, 1, 1]

# Expand mask to match the shape of slots: [4, 5, 9, 64]
        mask = mask.expand(-1, -1, slots.shape[2], slots.shape[3])
        masked_slots = masked_slots*mask  # Mask out the frames

        return masked_slots, mask_frame_level

    def forward(self, slots, mask=None):
        B, F, S, D_slot = slots.shape

        # Add positional embedding to slots
        slots = slots + self.pos_embed_temporal.expand(slots.shape)

        # Apply random masking on the F dimension
        slots, random_mask = self.random_masking(slots)

        slots = slots.permute(0, 2, 1, 3)  # (B, S, F, D_slot)
        slots = torch.flatten(slots, start_dim=0, end_dim=1)  # (B * S, F, D_slot)

        if mask is not None:
            mask = torch.logical_not(mask.to(torch.bool))  # (B, F)
            mask = mask.repeat_interleave(S, dim=0)  # (B * S, F)
        else:
            mask = torch.logical_not(random_mask.to(torch.bool))  # (B, F)
            mask = mask.repeat_interleave(S, dim=0)  # (B * S, F)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            slots = self.slot_transformer(slots, src_key_padding_mask=mask)  # (B * S, F, D_slot)
            # slots = self.slot_transformer(slots, )  # (B * S, F, D_slot)

            # slots = self.slot_transformer(slots )  # (B * S, F, D_slot)


        unblocked_slot_num = self.F

        slots = slots.view(B, S, unblocked_slot_num, D_slot)  # (B, S, F, D_slot)
        slots = slots.permute(0, 2, 1, 3)  # (B, F, S, D_slot)

        slot_t = slots[:, self.F // 2 + 1]  # (B, S, D_slot)

        return slot_t, slots
