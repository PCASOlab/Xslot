import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
def add_noise_to_latents(latent_vectors, noise_std=0.01):
    # Add Gaussian noise to the latent vectors
    noise = torch.randn_like(latent_vectors) * noise_std
    perturbed_latent_vectors = latent_vectors + noise
    return perturbed_latent_vectors
def cosine_similarity_matrix(vectors1, vectors2):
    # Normalize the vectors first
    vectors1 = F.normalize(vectors1, dim=-1)
    vectors2 = F.normalize(vectors2, dim=-1)
    
    # Compute pairwise cosine similarity (M x M) between vectors1 and vectors2
    cosine_sim_matrix = torch.matmul(vectors1, vectors2.transpose(-1, -2))  # (M, M)
    return cosine_sim_matrix

def find_best_permutation(cosine_sim_matrix):
    # Convert the cosine similarity matrix to a numpy array (for linear_sum_assignment)
    sim_matrix_np = cosine_sim_matrix.detach().cpu().numpy()
    
    # Use the Hungarian algorithm to find the best matching (maximizing similarity)
    row_indices, col_indices = linear_sum_assignment(-sim_matrix_np)  # Minimizing -cosine similarity
    return row_indices, col_indices

def permute_vectors(vectors, col_indices):
    # Permute the vectors according to the best matching indices
    return vectors[col_indices]

def align_permutations_across_frames(batch_frames_matrix):
    # Input shape: (Batch, Frames, M, N)
    batch_size, num_frames, M, N = batch_frames_matrix.size()
    
    # Iterate through each batch
    for batch_idx in range(batch_size):
        # For each frame, align it with the previous frame
        for frame_idx in range(1, num_frames):
            vectors_i = batch_frames_matrix[batch_idx, frame_idx - 1]  # Frame i (M, N)
            vectors_i1 = batch_frames_matrix[batch_idx, frame_idx]      # Frame i+1 (M, N)
            
            # Compute cosine similarity between frame i and frame i+1
            cosine_sim_matrix = cosine_similarity_matrix(vectors_i, vectors_i1)
            
            # Find the best permutation that maximizes cosine similarity
            row_indices, col_indices = find_best_permutation(cosine_sim_matrix)
            
            # Permute the vectors in frame i+1 based on the best matching
            batch_frames_matrix[batch_idx, frame_idx] = permute_vectors(vectors_i1, col_indices)
    
    return batch_frames_matrix
def align_permutations_across_n_frames(batch_frames_matrix,n=2):
    # Input shape: (Batch, Frames, M, N)
    batch_size, num_frames, M, N = batch_frames_matrix.size()
    
    # Iterate through each batch
    for batch_idx in range(batch_size):
        # For each frame, align it with the previous frame
        for frame_idx in range(2, num_frames):
            vectors_i = batch_frames_matrix[batch_idx, frame_idx - n]  # Frame i (M, N)
            vectors_i1 = batch_frames_matrix[batch_idx, frame_idx]      # Frame i+1 (M, N)
            
            # Compute cosine similarity between frame i and frame i+1
            cosine_sim_matrix = cosine_similarity_matrix(vectors_i, vectors_i1)
            
            # Find the best permutation that maximizes cosine similarity
            row_indices, col_indices = find_best_permutation(cosine_sim_matrix)
            
            # Permute the vectors in frame i+1 based on the best matching
            batch_frames_matrix[batch_idx, frame_idx] = permute_vectors(vectors_i1, col_indices)
    
    return batch_frames_matrix


def cosine_similarity_loss_neighboring(latent_vectors):
    """
    Computes a regularization loss based on the cosine similarity between
    neighboring frames' latent vectors.
    
    Args:
        latent_vectors: Tensor of shape (batch_size, num_frames, M, N), 
                        where M is the number of vectors per frame, 
                        and N is the dimensionality of each vector.
        num_frames: The number of frames in the batch.
        
    Returns:
        reg_loss: The regularization loss that penalizes dissimilarities 
                  between neighboring frames.
    """
    batch_size, num_frames, M, N = latent_vectors.size()
    batch_size = latent_vectors.size(0)
    reg_loss = 0.0

    # Loop over frames
    for i in range(num_frames - 1):
        # Get the latent vectors for the i-th and (i+1)-th frames
        vectors_i = latent_vectors[:, i, :, :]  # Shape: (batch_size, M, N)
        vectors_i_plus_1 = latent_vectors[:, i+1, :, :]  # Shape: (batch_size, M, N)

        # Compute cosine similarity between corresponding vectors of neighboring frames
        cosine_sim = F.cosine_similarity(vectors_i.unsqueeze(2), vectors_i_plus_1.unsqueeze(1), dim=-1)
        
        # Take the mean similarity across all vectors and across the batch
        reg_loss += (1 - cosine_sim.mean())

    reg_loss /= (num_frames - 1)  # Normalize over the number of frame pairs

    return reg_loss
def affinity_matrix_regularization(latent_vectors):
    """
    Computes a regularization loss using the affinity matrix (cosine similarity) 
    across all frames in the batch.

    Args:
        latent_vectors: Tensor of shape (batch_size, num_frames, M, N), where M 
                        is the number of vectors per frame, and N is the dimensionality 
                        of each vector.
        num_frames: Number of frames in the sequence.

    Returns:
        reg_loss: Affinity matrix-based regularization loss.
    """
    batch_size, num_frames, M, N = latent_vectors.size()

    reg_loss = 0.0

    for b in range(batch_size):
        for i in range(num_frames):
            vectors_i = latent_vectors[b, i, :, :]  # Shape: (M, N)
            for j in range(i + 1, num_frames):
                vectors_j = latent_vectors[b, j, :, :]  # Shape: (M, N)

                # Compute the cosine similarity between frame i and frame j
                cosine_sim = F.cosine_similarity(vectors_i.unsqueeze(1), vectors_j.unsqueeze(0), dim=-1)

                # Compute the affinity loss: higher cosine similarity should be favored
                reg_loss += (1 - cosine_sim.mean())

    reg_loss /= (batch_size * num_frames * (num_frames - 1) / 2)  # Normalize

    return reg_loss
# # Example usage:
# Batch, Frames, M, N = 2, 5, 10, 128  # Example sizes
# batch_frames_matrix = torch.randn(Batch, Frames, M, N)

# # Align permutations across frames based on cosine similarity
# aligned_matrix = align_permutations_across_frames(batch_frames_matrix)
def compute_cosine_similarity_matrix(latent_vectors):
    """
    Compute cosine similarity matrix between latent vectors across frames.
    
    Args:
        latent_vectors: Tensor of shape (batch_size, num_frames, M, N), where
                        M is the number of vectors per frame, and N is the 
                        dimensionality of each vector.

    Returns:
        affinity_matrix: Tensor of shape (num_frames, num_frames, M, M), where
                         affinity_matrix[i, j, m1, m2] represents the cosine 
                         similarity between vector m1 of frame i and vector m2 
                         of frame j.
    """
    batch_size, num_frames, M, N = latent_vectors.shape
    affinity_matrix = torch.zeros((batch_size, num_frames, num_frames, M, M))

    for b in range(batch_size):
        for i in range(num_frames):
            for j in range(num_frames):
                vec_i = latent_vectors[b, i, :, :]  # Shape: (M, N)
                vec_j = latent_vectors[b, j, :, :]  # Shape: (M, N)
                
                # Compute pairwise cosine similarity between vectors in frame i and frame j
                cos_sim = F.cosine_similarity(vec_i.unsqueeze(1), vec_j.unsqueeze(0), dim=-1)
                
                # Store cosine similarity in the affinity matrix
                affinity_matrix[b, i, j, :, :] = cos_sim

    return affinity_matrix
def sinkhorn(log_alpha, n_iters=20):
    """
    Sinkhorn algorithm for softmax-like matrix normalization (differentiable approximation of permutation).
    
    Args:
        log_alpha: Tensor of log-affinity matrix (similarity scores), shape [batch, M, M].
        n_iters: Number of Sinkhorn iterations.
    
    Returns:
        soft_perm_matrix: Soft permutation matrix after applying Sinkhorn iterations.
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)

def compute_soft_permutation(affinity_matrix, n_iters=20):
    """
    Computes soft permutation matrices using Sinkhorn iterations.
    
    Args:
        affinity_matrix: Tensor of shape (batch_size, num_frames, num_frames, M, M) representing cosine similarities.
        n_iters: Number of Sinkhorn iterations for approximation.
    
    Returns:
        soft_permutations: Tensor of soft permutation matrices, shape (batch_size, num_frames, num_frames, M, M).
    """
    batch_size, num_frames, _, M, _ = affinity_matrix.shape
    soft_permutations = torch.zeros((batch_size, num_frames, num_frames, M, M), device=affinity_matrix.device)

    for b in range(batch_size):
        for i in range(num_frames - 1):
            log_alpha = affinity_matrix[b, i, i+1, :, :]  # Affinity matrix between frame i and i+1
            # Apply Sinkhorn normalization to get a soft permutation matrix
            soft_perm_matrix = sinkhorn(log_alpha, n_iters=n_iters)
            soft_permutations[b, i, i+1, :, :] = soft_perm_matrix
    
    return soft_permutations
def find_optimal_permutation(affinity_matrix):
    """
    Find the optimal permutation that maximizes cosine similarity across frames
    using the Hungarian algorithm.

    Args:
        affinity_matrix: Tensor of shape (num_frames, num_frames, M, M), where
                         each element is the cosine similarity between vectors.

    Returns:
        permutations: A list of optimal permutations for each batch.
    """
    batch_size, num_frames, _, M, _ = affinity_matrix.shape
    permutations = []

    for b in range(batch_size):
        perm_per_frame = []
        for i in range(num_frames - 1):
            # Select the cosine similarity matrix between frame i and frame i+1
            similarity_matrix = affinity_matrix[b, i, i+1, :, :].detach().cpu().numpy()
            
            # Use Hungarian matching to find the best permutation
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Maximize similarity
            
            # Store the best permutation (col_ind gives the optimal assignment)
            perm_per_frame.append(col_ind)
        
        # Add the permutation sequence for the current batch
        permutations.append(perm_per_frame)

    return permutations

# def apply_permutation(latent_vectors, permutations):
#     """
#     Apply the optimal permutation to reorder the latent vectors across frames.

#     Args:
#         latent_vectors: Tensor of shape (batch_size, num_frames, M, N).
#         permutations: Optimal permutations for each frame.

#     Returns:
#         permuted_latent_vectors: Tensor with reordered latent vectors.
#     """
#     batch_size, num_frames, M, N = latent_vectors.shape
#     permuted_latent_vectors = latent_vectors.clone()
#     permutations = permutations.to(latent_vectors.device) 
#     for b in range(batch_size):
#         for i in range(num_frames - 1):
#             perm = permutations[b][i]
#             # Reorder vectors in frame i+1 based on the permutation
#             perm = perm.long()  # Convert the permutation matrix to long type
#             permuted_latent_vectors[b, i+1, :, :] = torch.gather(latent_vectors[b, i+1, :, :], 0, perm)

#     return permuted_latent_vectors
def apply_permutation(latent_vectors, permutations):
    """
    Apply the optimal permutation to reorder the latent vectors across frames.

    Args:
        latent_vectors: Tensor of shape (batch_size, num_frames, M, N).
        permutations: Optimal permutations for each frame.

    Returns:
        permuted_latent_vectors: Tensor with reordered latent vectors.
    """
    batch_size, num_frames, M, N = latent_vectors.shape
    permuted_latent_vectors = latent_vectors.clone()

    for b in range(batch_size):
        for i in range(num_frames - 1):
            perm = permutations[b][i]
            # Reorder vectors in frame i+1 based on the permutation
            permuted_latent_vectors[b, i+1, :, :] = latent_vectors[b, i+1, perm, :]

    return permuted_latent_vectors
