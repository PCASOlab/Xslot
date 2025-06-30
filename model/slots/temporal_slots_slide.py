import warnings

import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as f
from torch.nn import init
import random
import timm

def apply_sliding_window_avg(model, input_slots, window_size=5, overlap=3):
    B, F, S, D_slot = input_slots.shape
    stride = window_size - overlap  # Calculate the stride based on overlap (in this case, stride=3)
    
    assert F >= window_size, "Sequence length must be at least as large as the window size."
    assert stride > 0, "Stride must be positive to avoid infinite loop."
    
    # Initialize a tensor to accumulate results and a count matrix for averaging
    accumulated_slots = torch.zeros_like(input_slots)
    count_matrix = torch.zeros((B, F, S, D_slot), device=input_slots.device)
    
    # Slide over the frames with calculated stride
    for i in range(0, F - window_size + 1, stride):
        # Extract a window of frames
        window_slots = input_slots[:, i:i+window_size, :, :]
        # Apply Temporal_Binder model to get the slots for this window
        slotst,slots = model(window_slots)
        
        # Add the slots to the accumulated tensor and increment counts for averaging
        accumulated_slots[:, i:i+window_size, :, :] += slots
        count_matrix[:, i:i+window_size, :, :] += 1
    
    # Compute the averaged slots by dividing accumulated results by the count matrix
    averaged_slots = accumulated_slots / count_matrix
    return averaged_slots  # Shape: (B, F, S, D_slot)