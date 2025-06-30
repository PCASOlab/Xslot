from typing import Optional

import torch
from torch import nn
from video_SA.videosaur.modules import utils
from video_SA.videosaur.utils import config_as_kwargs, make_build_fn
from typing import Callable, List, Optional, Union
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import random
from working_dir_root import Evaluation_slots


@make_build_fn(__name__, "presence_nn")
def build(config, name: str,Sim_threshold=0.9):
    if name == "build_two_layer_mlp":
        
        return Presence_NN(config.input_dim, config.output_dim, config.hidden_dim, config.initial_layer_norm, config.residual)
    elif name == "sim_former":
        return Presence_NN_sim_former(input_dim=  config.input_dim,output_dim= config.output_dim,
                                      hidden_dim =  config.hidden_dim, initial_layer_norm= config.initial_layer_norm, residual = config.residual)
    elif name == "sim_cluster":
        return Presence_NN_sim_cluster(input_dim=  config.input_dim,output_dim= config.output_dim,
                                      hidden_dim =  config.hidden_dim, initial_layer_norm= config.initial_layer_norm, residual = config.residual)
    #  learnable slot merger
    elif name == "sim_merger":
        return Presence_NN_sim_merger(input_dim=  config.input_dim,output_dim= config.output_dim,
                                      hidden_dim =  config.hidden_dim, initial_layer_norm= config.initial_layer_norm, residual = config.residual,Sim_threshold=Sim_threshold)




class Presence_NN(nn.Module):
     

    def __init__(
        self,
        input_dim, output_dim, hidden_dim, initial_layer_norm: bool = False, residual: bool = False):
        super().__init__()
        self. presence_nn =  build_two_layer_mlp(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, initial_layer_norm=initial_layer_norm, residual =residual )

    def forward(self, slots: torch.Tensor, global_step = None):
        b, k, _ = slots.shape
        prev_decision = torch.ones(b, k, dtype=slots.dtype, device=slots.device) #prev_decision [b, k]
        
        slots_keep_prob = self.presence_nn(slots) #slots_keep_prob [b, k, 2]
        
        tau = 1
        
        current_keep_decision = F.gumbel_softmax(slots_keep_prob, hard=True, tau = tau)[...,1]
        hard_keep_decision = current_keep_decision * prev_decision #hard_keep_decision [b, k]
        slots_keep_prob = F.softmax(slots_keep_prob, dim = -1)[...,1]
        # hard_idx[...,0]
        return   hard_keep_decision
class Presence_NN_sim_cluster(nn.Module):
    def __init__(
        self,
        input_dim,  # Dimension of each slot vector
        output_dim,  # Output dimension for slot keep decision
        hidden_dim,  # List of hidden dimensions for layersster
        initial_layer_norm: bool = False,
        n_heads: int = 3,  # Number of attention heads in the transformer
        n_layers: int = 2,  # Number of transformer layers
        residual: bool = False,
        K=9
    ):
        super().__init__()

        # MLP for processing individual slot vectors
        self.slot_mlp =  self.build_mlp(input_dim, hidden_dim, initial_layer_norm, residual)
        # self. slot_merger =  Slot_Merger_Cosine()
        self. slot_merger =  Slot_Merger()

        # Define the Transformer Encoder for similarity vectors
        encoder_layer = nn.TransformerEncoderLayer(
            K,  # K: embedding size
            nhead=n_heads,  # Number of attention heads
            dim_feedforward=K * 12,  # Feedforward hidden size
            batch_first=True  # Input format: [batch, seq, embedding]
        )
        self.merge_threshold = nn.Parameter(torch.tensor(1.1), requires_grad=True)
        # Define TransformerEncoder
       
        self.similarity_transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )
 
        self.decision_mlp=build_two_layer_mlp (input_dim=K, output_dim=2, hidden_dim=hidden_dim, initial_layer_norm=initial_layer_norm, residual =residual)
        
    def build_mlp(self, input_dim, hidden_dim_list, initial_layer_norm, residual):
        """Helper function to build an MLP with varying hidden dimensions."""
        layers = []
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Linear(input_dim, hidden_dim_list[i]))
            if initial_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim_list[i]))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim_list[i]  # Update input_dim for next layer
        return nn.Sequential(*layers)
    def forward(self, slots: torch.Tensor):
        """
        slots: Tensor of shape [batch_size, K_number_of_slots, input_dim]
        """
        b, k, d = slots.shape

        # new_slots, current_keep_decision = self.slot_merger (slots,self.merge_threshold )
        new_slots, current_keep_decision = self.slot_merger (slots  )


 
        return new_slots,current_keep_decision


class Presence_NN_sim_merger(nn.Module):
    def __init__(
        self,
        input_dim,  # Dimension of each slot vector
        output_dim,  # Output dimension for slot keep decision
        hidden_dim,  # List of hidden dimensions for layersster
        initial_layer_norm: bool = False,
        n_heads: int = 3,  # Number of attention heads in the transformer
        n_layers: int = 2,  # Number of transformer layers
        residual: bool = False,
        K=9,
        Sim_threshold = 0.9
    ):
        super().__init__()

        # MLP for processing individual slot vectors
        self.slot_mlp =  self.build_mlp(input_dim, hidden_dim, initial_layer_norm, residual)
        # self. slot_merger =  Slot_Merger_Cosine()
        self. slot_merger =  Slot_Merger_Cosine(Sim_threshold=Sim_threshold)

        # Define the Transformer Encoder for similarity vectors
        encoder_layer = nn.TransformerEncoderLayer(
            K,  # K: embedding size
            nhead=n_heads,  # Number of attention heads
            dim_feedforward=K * 12,  # Feedforward hidden size
            batch_first=True  # Input format: [batch, seq, embedding]
        )
        self.merge_threshold = nn.Parameter(torch.tensor(1.1), requires_grad=True)
        # Define TransformerEncoder
       
        self.similarity_transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )
 
        self.decision_mlp=build_two_layer_mlp (input_dim=K, output_dim=2, hidden_dim=hidden_dim, initial_layer_norm=initial_layer_norm, residual =residual)
        
    def build_mlp(self, input_dim, hidden_dim_list, initial_layer_norm, residual):
        """Helper function to build an MLP with varying hidden dimensions."""
        layers = []
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Linear(input_dim, hidden_dim_list[i]))
            if initial_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim_list[i]))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim_list[i]  # Update input_dim for next layer
        return nn.Sequential(*layers)
    def forward(self, slots: torch.Tensor):
        """
        slots: Tensor of shape [batch_size, K_number_of_slots, input_dim]
        """
        b, k, d = slots.shape

        # new_slots, current_keep_decision = self.slot_merger (slots,self.merge_threshold )
        new_slots, current_keep_decision = self.slot_merger (slots  )


 
        return new_slots,current_keep_decision



class Slot_Merger_Cosine(nn.Module):
    def __init__(self, Sim_threshold =0.9):
        super().__init__()
        self.num_slots = 9
        self.slot_dim = 64
        self.cluster_drop_p = 0.15
        self.Sim_threshold = Sim_threshold
         # Learnable threshold for cosine similarity
        self.epsilon = 1e-8

    def cosine_similarity(self, slots):
        # Calculate cosine similarity matrix (B, S, S)
        norm = torch.norm(slots, dim=-1, keepdim=True)  # (B, S, 1)
        similarity_matrix = torch.matmul(slots, slots.transpose(1, 2)) / (norm * norm.transpose(1, 2) + self.epsilon)  # (B, S, S)
        return similarity_matrix

    def forward(self, slots):
        B, S, D = slots.shape  # B: batch size, S: number of slots, D: slot dimension
        
        # Step 1: Compute the cosine similarity between all slot pairs
        similarity_matrix = self.cosine_similarity(slots)  # (B, S, S)

        # Step 2: Compare similarity with learnable threshold and create merge mask
        if random.random() > self.cluster_drop_p or Evaluation_slots==True:
            merge_mask = (similarity_matrix > self.Sim_threshold).float()  
            
            # merge_mask = torch.sigmoid(similarity_matrix - merge_threshold)  # (B, S, S), 1 if merge, 0 otherwise
        else:
            merge_mask = (similarity_matrix >  0.98).float()  
        # Step 3: Create slot assignment matrix
        slot_assignment_nums = torch.sum(merge_mask, dim=-1, keepdim=True)  # (B, S, 1)
        
        # Step 4: Perform slot merging (averaging slots within each cluster)
        new_slots = torch.einsum('bij,bjd->bid', merge_mask, slots)  # (B, S, D)
        
        # Normalize the new slots (sum across merged slots)
        new_slots = new_slots / (slot_assignment_nums + self.epsilon)  # (B, S, D)

        # Step 5: Create slot_mask to indicate valid (non-zero) slots
        slot_mask = (slot_assignment_nums > 0).float()  # (B, S), 1 if valid slot, 0 if zero (merged/removed)
        # slot_mask = torch.ones_like(slot_assignment_nums).squeeze(-1)  # (B, S)
        # Step 6: Maintain permutation of unmerged slots
        unmerged_slots_mask = (slot_assignment_nums == 1).float()  # (B, S), 1 if slot was not merged
        for b in range(B):
            for s in range(S):
                if slot_assignment_nums[b, s] > 1:  # If the slot is part of a merged cluster
                    merge_indices = torch.nonzero(merge_mask[b, s]).squeeze()
                    if merge_indices.ndimension() > 0 and merge_indices.shape[0] > 1:
                        # Mask out the higher index in the merged pair
                        slot_mask[b, merge_indices[1:]] = 0  # Set mask to 0 for higher indices

        # Step 7: Maintain the merged slots at the lower index
        # Create a tensor to store the final slots with unmerged slots remaining at their positions
        final_slots = torch.zeros_like(slots)  # Initialize final slots to zero (same shape as slots)

        # Use the merge_mask to place the averaged merged slots at lower indices
        for b in range(B):
            for s in range(S):
                if slot_assignment_nums[b, s] > 0:
                    # If the slot is merged, assign the averaged value to the lower index position
                    merge_indices = torch.nonzero(merge_mask[b, s]).squeeze()
                    # Replace len(merge_indices) > 1 with shape checking
                    # Check if merge_indices contains more than one element
                    if merge_indices.ndimension() > 0 and merge_indices.shape[0] > 1:
                        # Compute the average of the merged slots
                        final_slots[b, merge_indices.min(), :] = new_slots[b, s, :]
                    else:
                        # If no merging, copy the original slot
                        final_slots[b, s, :] = slots[b, s, :]
                else:
                    # If not merged, retain the slot at its index
                    final_slots[b, s, :] = slots[b, s, :]

        # Step 8: Apply unmerged slots mask to retain unmerged slots' values
        # Ensure unmerged_slots_mask_expanded has the same shape as final_slots (B, S, D)
        # unmerged_slots_mask_expanded = unmerged_slots_mask.unsqueeze(-1)  # Shape (B, S, 1)

        # # Broadcast unmerged_slots_mask_expanded to match final_slots shape (B, S, D)
        # unmerged_slots_mask_expanded = unmerged_slots_mask_expanded.expand(-1, -1, final_slots.shape[-1])  # Shape (B, S, D)

        # # Perform the element-wise multiplication
        # final_slots = final_slots * unmerged_slots_mask_expanded  # Shape (B, S, D)

        return final_slots, slot_mask.reshape(B,S)
    

class Slot_Merger_Cosine_avg(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_slots = 9
        self.slot_dim = 64
        self.cluster_drop_p = 0.15
        self.epsilon = 1e-8  # Learnable threshold for cosine similarity

    def cosine_similarity(self, slots):
        # Calculate cosine similarity matrix (B, S, S)
        norm = torch.norm(slots, dim=-1, keepdim=True)  # (B, S, 1)
        similarity_matrix = torch.matmul(slots, slots.transpose(1, 2)) / (norm * norm.transpose(1, 2) + self.epsilon)  # (B, S, S)
        return similarity_matrix

    def forward(self, slots):
        B, S, D = slots.shape  # B: batch size, S: number of slots, D: slot dimension
        
        # Step 1: Compute the cosine similarity between all slot pairs
        similarity_matrix = self.cosine_similarity(slots)  # (B, S, S)

        # Step 2: Compare similarity with learnable threshold and create merge mask
        if random.random() > self.cluster_drop_p or Evaluation_slots==True:
            merge_mask = (similarity_matrix > 0.9).float()  
        else:
            merge_mask = (similarity_matrix > 0.98).float()  

        # Step 3: Create slot assignment matrix
        slot_assignment_nums = torch.sum(merge_mask, dim=-1, keepdim=True)  # (B, S, 1)
        
        # Step 4: Perform slot merging (averaging slots within each cluster)
        new_slots = torch.einsum('bij,bjd->bid', merge_mask, slots)  # (B, S, D)
        
        # Normalize the new slots (sum across merged slots)
        new_slots = new_slots / (slot_assignment_nums + self.epsilon)  # (B, S, D)

        # Step 5: Create slot_mask to indicate valid (non-zero) slots
        # Initialize slot_mask as 1 for all slots
        slot_mask = torch.ones_like(slot_assignment_nums).squeeze(-1)  # (B, S)

        # Step 6: Mask out the higher index in merged pairs (e.g., slot 3 if slots 2 and 3 are merged)
        for b in range(B):
            for s in range(S):
                if slot_assignment_nums[b, s] > 1:  # If the slot is part of a merged cluster
                    merge_indices = torch.nonzero(merge_mask[b, s]).squeeze()
                    if merge_indices.ndimension() > 0 and merge_indices.shape[0] > 1:
                        # Mask out the higher index in the merged pair
                        slot_mask[b, merge_indices[1:]] = 0  # Set mask to 0 for higher indices

        # Step 7: Assign the averaged value to all merged slots
        final_slots = torch.zeros_like(slots)  # Initialize final slots to zero (same shape as slots)

        for b in range(B):
            for s in range(S):
                if slot_assignment_nums[b, s] > 0:
                    # If the slot is merged, assign the averaged value to all merged slots
                    merge_indices = torch.nonzero(merge_mask[b, s]).squeeze()
                    if merge_indices.ndimension() > 0 and merge_indices.shape[0] > 1:
                        # Assign the averaged value to all merged slots
                        final_slots[b, merge_indices, :] = new_slots[b, s, :]
                    else:
                        # If no merging, copy the original slot
                        final_slots[b, s, :] = slots[b, s, :]
                else:
                    # If not merged, retain the slot at its index
                    final_slots[b, s, :] = slots[b, s, :]

        return final_slots, slot_mask.reshape(B, S)


class Presence_NN_sim_former(nn.Module):
    def __init__(
        self,
        input_dim,  # Dimension of each slot vector
        output_dim,  # Output dimension for slot keep decision
        hidden_dim,  # List of hidden dimensions for layers
        initial_layer_norm: bool = False,
        n_heads: int = 3,  # Number of attention heads in the transformer
        n_layers: int = 2,  # Number of transformer layers
        residual: bool = False,
        K=9
    ):
        super().__init__()

        # MLP for processing individual slot vectors
        self.slot_mlp =  self.build_mlp(input_dim, hidden_dim, initial_layer_norm, residual)

        # Define the Transformer Encoder for similarity vectors
        encoder_layer = nn.TransformerEncoderLayer(
            K,  # K: embedding size
            nhead=n_heads,  # Number of attention heads
            dim_feedforward=K * 12,  # Feedforward hidden size
            batch_first=True  # Input format: [batch, seq, embedding]
        )
        
        # Define TransformerEncoder
       
        self.similarity_transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        # Final decision MLP
        # self.decision_mlp = self.build_mlp(
        #     hidden_dim[-1] + hidden_dim[0],  # Fused dimension (slot + similarity)
        #     [hidden_dim[-1], output_dim],  # Hidden layers to output decision
        #     initial_layer_norm,
        #     residual
        # )
        self.decision_mlp=build_two_layer_mlp (input_dim=K, output_dim=2, hidden_dim=hidden_dim, initial_layer_norm=initial_layer_norm, residual =residual)
        # self.decision_mlp = self.build_mlp(
        #    K,  # Fused dimension (slot + similarity)
        #     [hidden_dim[-1], output_dim],  # Hidden layers to output decision
        #     initial_layer_norm,
        #     residual
        # )
    def build_mlp(self, input_dim, hidden_dim_list, initial_layer_norm, residual):
        """Helper function to build an MLP with varying hidden dimensions."""
        layers = []
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Linear(input_dim, hidden_dim_list[i]))
            if initial_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim_list[i]))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim_list[i]  # Update input_dim for next layer
        return nn.Sequential(*layers)
    def forward(self, slots: torch.Tensor):
        """
        slots: Tensor of shape [batch_size, K_number_of_slots, input_dim]
        """
        b, k, d = slots.shape

        # Step 1: Compute pairwise cosine similarity matrix
        normalized_slots = F.normalize(slots, dim=-1)  # Normalize slot vectors
        similarity_matrix = torch.matmul(normalized_slots, normalized_slots.transpose(1, 2))  # [b, k, k]

        # Step 2: Remove self-similarity (subtract diagonal)
        identity_matrix = torch.eye(k, device=slots.device).unsqueeze(0).expand(b, -1, -1)  # [b, k, k]
        similarity_matrix = similarity_matrix - similarity_matrix * identity_matrix  # Subtract diagonal

        # Step 3: Process slot features using slot_mlp
        slot_features = self.slot_mlp(slots)  # [b, k, hidden_dim[0]]

        # Step 4: Enhance similarity vectors using a transformer
        similarity_features = self.similarity_transformer(similarity_matrix)  # [b, k, hidden_dim[0]]

        # Step 5: Fuse slot and similarity features
        fused_features = torch.cat([slot_features, similarity_features], dim=-1)  # [b, k, hidden_dim[-1] + hidden_dim[0]]

        # Step 6: Make keep/discard decisions
        slot_keep_prob = self.decision_mlp(similarity_features)  # [b, k, output_dim]
        current_keep_decision = F.gumbel_softmax(slot_keep_prob, hard=True, tau=1)[..., 1]  # [b, k]

        return current_keep_decision
class Slot_Merger(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.num_slots = 9
        self.slot_dim = 64
        self.cluster_drop_p = 0.1
        self.slot_merge_coeff = 0.15# higher tend to merge 
        self.epsilon = 1e-4

    def forward(self, slots):
        # :arg slots: (B, S, D_slot)
        # :arg patch_attn: (B, S, N')
        #
        # :return new_slots: (B, S, D_slot)
        # :return new_patch_attn: (B, S, N')
        # :return slot_nums: (B)

        assert torch.sum(torch.isnan(slots)) == 0
     
        B, S, D = slots.shape
        slots_np = slots.detach().cpu().numpy()  # (B, S, D_slot)

        # Initialize new_slots and clustering labels
        new_slots = torch.zeros_like(slots)  # (B, S, D_slot)
        slot_masks = torch.zeros((B, S), dtype=torch.bool, device=slots.device)  # (B, S) to track kept slots

        for i in range(B):
            # Decide whether to perform clustering
            if random.random() < self.cluster_drop_p:
                # Skip clustering, assign each slot to itself
                clusters = np.arange(self.num_slots)
            else:
                # Perform clustering
                AC = AgglomerativeClustering(
                    n_clusters=None, 
                    metric="cosine", 
                    compute_full_tree=True, 
                    distance_threshold=self.slot_merge_coeff, 
                    linkage="complete"
                )
                AC.fit(slots_np[i])
                clusters = AC.labels_

                # If all slots are merged into one cluster, skip clustering
                if clusters.max() <= 1:
                    clusters = np.arange(self.num_slots)

            # Maintain permutation consistency
            merged_indices = set()
            for cluster_id in range(clusters.max() + 1):
                cluster_mask = clusters == cluster_id
                if cluster_mask.sum() > 1:  # If the cluster has more than one slot
                    merged_indices.add(cluster_id)
                    # Merge the slots in this cluster
                    merged_slot = slots[i, cluster_mask].mean(dim=0)
                    new_slots[i, len(merged_indices) - 1] = merged_slot
                    slot_masks[i, len(merged_indices) - 1] = True
                else:
                    # Keep the unmerged slot in its original index
                    original_idx = np.where(cluster_mask)[0][0]
                    new_slots[i, original_idx] = slots[i, original_idx]
                    slot_masks[i, original_idx] = True

        # Replace unused slots with zeros
        for i in range(B):
            unused_slots = torch.where(~slot_masks[i])[0]
            new_slots[i, unused_slots] = 0
 

        return new_slots, slot_masks.float()

def build_two_layer_mlp(
    input_dim, output_dim, hidden_dim, initial_layer_norm: bool = False, residual: bool = False
):
    """Build a two layer MLP, with optional initial layer norm.

    Separate class as this type of construction is used very often for slot attention and
    transformers.
    """
    return build_mlp(
        input_dim, output_dim, hidden_dim, initial_layer_norm=initial_layer_norm, residual=residual
    )
def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


def get_activation_fn(name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = None):
    if callable(name):
        return name

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "relu_squared":
        return ReLUSquared(inplace=inplace)
    elif name == "leaky_relu":
        if leaky_relu_slope is None:
            raise ValueError("Slope of leaky ReLU was not defined")
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {name}")

class ReLUSquared(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu(x, inplace=self.inplace) ** 2
    

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)