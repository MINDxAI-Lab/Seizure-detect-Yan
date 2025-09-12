import numpy as np
import scipy.io as sio
import os

elec_pos_path = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/scripts/elec_pos.npy"
save_path = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/scripts/adj_mat.mat"

elec_pos = np.load(elec_pos_path)  # shape: [19, 3]

# Compute pairwise Euclidean distances
dist_mat = np.linalg.norm(elec_pos[:, None, :] - elec_pos[None, :, :], axis=-1)

# Convert distances to adjacency using Gaussian kernel
sigma = np.mean(dist_mat)
adj_mat = np.exp(-dist_mat**2 / (2 * sigma**2))

# Save the adjacency matrix in .mat format
sio.savemat(save_path, {'adj_mat': adj_mat})
