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
    def __init__(self, slot_dim, num_slots,num_frame):
        super().__init__()
        
        self.slot_dim =  slot_dim
        self.num_slots =  num_slots
        # self.N = args.N
        self.F = num_frame

        encoder_layer = nn.TransformerEncoderLayer(self.slot_dim, nhead=8, dim_feedforward=4*self.slot_dim, batch_first=True)
        self.slot_transformer = nn.TransformerEncoder(encoder_layer, 3)

        self.pos_embed_temporal = nn.Parameter(torch.Tensor(1, self.F, 1, self.slot_dim))
        init.normal_(self.pos_embed_temporal, mean=0., std=.02)

    def forward(self, slots, mask=None,usingmask=False):
        # :arg slots: (B * F, S, D_slot)
        # :arg mask: (B, F)
        #
        # :return: (B * F, S, D_slot)

        B,F, S, D_slot = slots.shape

        # slots = slots.view(-1, self.F, S, D_slot)                       # (B, F, S, D_slot)
        slots = slots + self.pos_embed_temporal.expand(slots.shape)

        B = slots.shape[0]

        slots = slots.permute(0, 2, 1, 3)                                   # (B, S, F, D_slot)
        slots = torch.flatten(slots, start_dim=0, end_dim=1)                # (B * S, F, D_slot)

        if mask is not None:
            mask = torch.logical_not(mask.to(torch.bool))                       # (B, F)
            mask = mask.repeat_interleave(S, dim=0)                             # (B * S, F)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            if mask is not None:
                slots = self.slot_transformer(slots, src_key_padding_mask=mask) # (B * S, F, D_slot)
            else:
                slots = self.slot_transformer(slots, src_key_padding_mask=mask) # (B * S, F, D_slot)

        # unblocked_slot_num = (torch.mean(mask.float(), dim=0) != 1).sum().long()
        unblocked_slot_num = self.F
        
        slots = slots.view(B, S, unblocked_slot_num, D_slot)                # (B, S, F, D_slot)
        slots = slots.permute(0, 2, 1, 3)                                   # (B, F, S, D_slot)

        slot_t = slots[:, self.F//2+1]                                           # (B, S, D_slot)

        return slot_t, slots
