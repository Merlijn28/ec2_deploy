"""
CPU-based acceleration utilities for mesh generation
Provides optimized CPU operations for mesh generation
"""

import numpy as np
from scipy import ndimage


def bilinear_interpolation(
    image: np.ndarray,
    uv_coords: np.ndarray,
    mode: str = 'wrap'
) -> np.ndarray:
    """
    CPU-based bilinear interpolation for image sampling
    
    Args:
        image: Grayscale image array (H, W) normalized to [0, 1]
        uv_coords: UV coordinates (N, 2) in range [0, 1]
        mode: Border mode ('wrap', 'clamp', or 'reflect')
        
    Returns:
        np.ndarray: Sampled values for each UV coordinate
    """
    height, width = image.shape
    u = (uv_coords[:, 0] % 1.0) * width
    v = (uv_coords[:, 1] % 1.0) * height
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    return ndimage.map_coordinates(image, [v, u], order=1, mode='wrap')


def uv_computation(vertices: np.ndarray) -> np.ndarray:
    """
    CPU-based UV coordinate computation using cylindrical mapping
    
    Args:
        vertices: Vertex positions (N, 3) - side surface vertices only
        
    Returns:
        np.ndarray: UV coordinates (N, 2) in range [0, 1]
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    angle = np.arctan2(y, x)
    u = (angle + np.pi) / (2 * np.pi)
    
    z_min = np.min(z)
    z_max = np.max(z)
    z_range = z_max - z_min
    if z_range > 0.001:
        v = (z - z_min) / z_range
        v = np.clip(v, 0.0, 1.0)
        v = 1.0 - v
    else:
        v = np.ones_like(z)
    
    return np.column_stack([u, v])


def identify_outer_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    CPU-based identification of outer surface vertices
    Identifies vertices with maximum radius at each height level.
    
    Args:
        vertices: Vertex positions (N, 3)
        
    Returns:
        np.ndarray: Boolean mask indicating outer vertices
    """
    radius = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    z = vertices[:, 2]
    z_rounded = np.round(z, decimals=1)
    z_unique = np.unique(z_rounded)
    
    outer_mask = np.zeros(len(vertices), dtype=bool)
    
    for z_level in z_unique:
        height_mask = z_rounded == z_level
        if np.sum(height_mask) == 0:
            continue
        
        radii_at_height = radius[height_mask]
        if len(radii_at_height) == 0:
            continue
        
        max_radius = np.max(radii_at_height)
        min_radius = np.min(radii_at_height)
        radius_range = max_radius - min_radius
        
        if radius_range > 1.0:
            threshold = max_radius - (radius_range * 0.1)
            level_outer_mask = height_mask & (radius >= threshold)
        else:
            level_outer_mask = height_mask
        
        outer_mask |= level_outer_mask
    
    if np.sum(outer_mask) == 0:
        median_radius = np.median(radius)
        outer_mask = radius >= median_radius
    
    return outer_mask

