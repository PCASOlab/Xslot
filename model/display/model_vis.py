import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from visdom import Visdom

# Initialize Visdom
viz = Visdom(port=8097)
 
def plot_all_frames_on_projection_vizdom_2d(batch_frames_matrix):
    """
    Visualize latent vectors from all frames on a 2D plane using Visdom.

    Parameters:
        batch_frames_matrix: Tensor of shape (Batch, Frames, M, N)
    """
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = reduced_vectors[:, 0]
    Y = reduced_vectors[:, 1]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "PCA 1",
        "ylabel": "PCA 2",
        "title": f"Latent Vectors in 2D Plane (All Frames Combined)"
    }
    viz.scatter(
        X=np.column_stack((X, Y)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
 
def plot_all_frames_on_hypersphere_vizdom2(batch_frames_matrix):
    """
    Visualize latent vectors from all frames on the same hypersphere using Visdom.

    Parameters:
        batch_frames_matrix: Tensor of shape (Batch, Frames, M, N)
    """
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 3D using PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Project onto a unit sphere
    norms = np.linalg.norm(reduced_vectors, axis=1, keepdims=True)
    sphere_vectors = reduced_vectors / norms  # Normalize to project onto the sphere

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = sphere_vectors[:, 0]
    Y = sphere_vectors[:, 1]
    Z = sphere_vectors[:, 2]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "Sphere X",
        "ylabel": "Sphere Y",
        "zlabel": "Sphere Z",
        "title": "Latent Vectors on a 3D Sphere (All Frames Combined)"
    }
    viz.scatter(
        X=np.column_stack((X, Y, Z)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
def plot_all_frames_on_projection_vizdom_2d_tsne(batch_frames_matrix):
    """
    Visualize latent vectors from all frames on a 2D plane using t-SNE and Visdom.

    Parameters:
        batch_frames_matrix: Tensor of shape (Batch, Frames, M, N)
    """
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_vectors = tsne.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = reduced_vectors[:, 0]
    Y = reduced_vectors[:, 1]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "t-SNE Dim 1",
        "ylabel": "t-SNE Dim 2",
        "title": "Latent Vectors in 2D (t-SNE)"
    }
    viz.scatter(
        X=np.column_stack((X, Y)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
def plot_all_frames_on_hypersphere_vizdom_tsne(batch_frames_matrix):
    
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 3D using t-SNE
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    reduced_vectors = tsne.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Project onto a unit sphere
    norms = np.linalg.norm(reduced_vectors, axis=1, keepdims=True)
    sphere_vectors = reduced_vectors / norms  # Normalize to project onto the sphere

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = sphere_vectors[:, 0]
    Y = sphere_vectors[:, 1]
    Z = sphere_vectors[:, 2]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "t-SNE Dim 1",
        "ylabel": "t-SNE Dim 2",
        "zlabel": "t-SNE Dim 3",
        "title": "Latent Vectors on 3D Sphere (t-SNE)"
    }
    viz.scatter(
        X=np.column_stack((X, Y, Z)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
if __name__ == '__main__':
    # Example usage
    Batch, Frames, M, N = 1, 5, 10, 50  # One batch, 5 frames, 10 vectors per frame, vector size 50
    batch_frames_matrix = torch.randn(Batch, Frames, M, N)

    # Plot all frames' vectors on the same hypersphere
    plot_all_frames_on_hypersphere_vizdom_tsne(batch_frames_matrix)