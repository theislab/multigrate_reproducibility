import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def find_closest_points(adata_rna, adata_msi):
    """
    Find the closest points between two AnnData objects based on their spatial coordinates.

    Parameters:
    adata_rna (AnnData): AnnData object containing RNA data with spatial coordinates in 'spatial_warp'.
    adata_msi (AnnData): AnnData object containing MSI data with spatial coordinates in 'spatial_warp'.

    Returns:
    tuple: A tuple containing:
        - matching_df (DataFrame): DataFrame showing the matching pairs of points and their distances.
        - matching_rna (ndarray): Array of RNA data corresponding to the closest points.
        - matching_msi (ndarray): Array of MSI data.
    """
    # Use the 'spatial_warp' coordinates
    adata_rna_coords = adata_rna.obsm['spatial_warp']
    adata_msi_coords = adata_msi.obsm['spatial_warp']

    # Step 1: Build a spatial tree for `adata_rna`
    tree_adata_rna = cKDTree(adata_rna_coords)

    # Step 2: Query the tree to find the closest point in `adata_rna` for each point in `adata_msi`
    distances, indices = tree_adata_rna.query(adata_msi_coords)

    # Step 3: Create a DataFrame to show the matching pairs
    matching_df = pd.DataFrame({
        'adata_msi_index': np.arange(len(adata_msi_coords)),
        'adata_msi_x': adata_msi_coords[:, 0],
        'adata_msi_y': adata_msi_coords[:, 1],
        'closest_adata_rna_index': indices,
        'adata_rna_x': adata_rna_coords[indices, 0],
        'adata_rna_y': adata_rna_coords[indices, 1],
        'distance': distances
    })

    # Extract the matching RNA and MSI data
    matching_rna = adata_rna.X[indices]
    matching_msi = adata_msi.X

    return matching_df, matching_rna.toarray(), matching_msi


