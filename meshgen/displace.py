"""
Displacement Mapping
Applies image-based displacement to mesh vertices
"""

import numpy as np
from PIL import Image
import trimesh
from scipy import ndimage
import io
import base64
import gc

# Import CPU acceleration utilities
from meshgen.accel import (
    bilinear_interpolation,
    uv_computation,
    identify_outer_vertices
)


def calculate_lamp_resolution(
    height: float,
    diameter: float,
    mm_per_pixel: float = 0.25,
    thickness: float = 3.0,  # in mm
    brim_diameter: float = 4.0,
    brim_height: float = 4.0,
    brim_overhang_angle: float = 45.0,
    top_brim: bool = True,
    bottom_brim: bool = True
) -> tuple:
    """
    Calculate the target image resolution (width, height) to match the lamp's grid resolution.
    
    Args:
        height: Height of the tube (in mm)
        diameter: Outer diameter of the lamp tube (in mm)
        mm_per_pixel: Millimeters per pixel for calculating mesh resolution
        thickness: Wall thickness (in mm)
        brim_diameter: Diameter extension for circular brims (in mm)
        brim_height: Vertical height/thickness of the brims (in mm)
        brim_overhang_angle: Overhang angle of brims in degrees
        top_brim: Include top brim height/slope in the calculation
        bottom_brim: Include bottom brim height/slope in the calculation
        
    Returns:
        tuple: (target_width, target_height) in pixels
    """
    if diameter is None:
        raise ValueError("diameter must be provided")

    radius_mm = float(diameter) / 2.0
    brim_radius_mm = float(brim_diameter) / 2.0

    # All values are now in mm
    height_mm = float(height)
    thickness_mm = float(thickness)  # Already in mm
    brim_height_mm = float(brim_height)
    
    # Convert overhang angle to radians
    overhang_angle_rad = np.deg2rad(brim_overhang_angle)
    
    # Calculate tube surface boundaries (excluding brims)
    tube_bottom_z = 0.0
    tube_top_z = height_mm
    tube_height_mm = tube_top_z - tube_bottom_z
    
    outer_tube_bottom_z = tube_bottom_z
    outer_tube_top_z = tube_top_z

    # Calculate where the tube surface should actually start and end (excluding brim overlap)
    if brim_radius_mm > 0:
        vertical_slope = brim_radius_mm * np.tan(overhang_angle_rad)
        has_top_brim = bool(top_brim)
        has_bottom_brim = bool(bottom_brim)

        if brim_height_mm > 0:
            if has_top_brim:
                outer_tube_top_z = tube_top_z - vertical_slope - brim_height_mm
            if has_bottom_brim:
                outer_tube_bottom_z = tube_bottom_z + vertical_slope + brim_height_mm
        elif overhang_angle_rad >= 1e-6:
            if has_top_brim:
                outer_tube_top_z = tube_top_z - vertical_slope
            if has_bottom_brim:
                outer_tube_bottom_z = tube_bottom_z + vertical_slope
    
    # Calculate surface dimensions
    outer_tube_height_mm = outer_tube_top_z - outer_tube_bottom_z
    circumference_mm = 2 * np.pi * radius_mm
    segments = max(3, int(np.ceil(circumference_mm / mm_per_pixel)))
    
    # Calculate vertical grid cells based on mm_per_pixel
    grid_cells = max(1, int(np.ceil(outer_tube_height_mm / mm_per_pixel)))
    
    # Ensure square cells by adjusting segments if needed
    ideal_segments = max(3, int(np.ceil((circumference_mm * grid_cells) / max(outer_tube_height_mm, 0.1))))
    segments = max(segments, ideal_segments)
    
    # Target image resolution: match the grid resolution
    target_width = segments  # Circumferential resolution
    target_height = grid_cells  # Vertical grid cells
    
    return (target_width, target_height)


def apply_displacement(
    mesh: trimesh.Trimesh,
    image_path: str,
    intensity: float = 0.5,
    scale: float = None,  # If None, will auto-calculate to fit surface
    offset: float = 0.0,
    target_width: int = None,  # Target image width to match lamp resolution
    target_height: int = None,  # Target image height to match lamp resolution
    thickness_mm: float = None,  # Wall thickness (mm) to constrain inward displacement
    min_clearance_mm: float = 0.6,  # Minimum space to keep between inner and displaced outer (mm)
    smoothing: float = 0.0,  # Gaussian blur sigma value (0 = no smoothing)
    progress_callback=None  # Optional callback(progress_pct, message) for progress updates
) -> tuple:
    """
    Apply image-based displacement to mesh vertices (outer surface only)
    Only applies to the cell grid on the outer surface, not to brims or inside.
    Image is scaled to fit the surface area without stretching (maintains aspect ratio).
    If target_width and target_height are provided, the image will be resampled to match
    the lamp's grid resolution.
    
    Args:
        mesh: Input mesh to displace
        image_path: Path to the displacement image (grayscale)
        intensity: Displacement intensity multiplier
        scale: Scale factor for UV coordinates (if None, auto-calculates to fit surface)
        offset: Base offset for displacement
        target_width: Target image width in pixels to match lamp resolution (optional)
        target_height: Target image height in pixels to match lamp resolution (optional)
        smoothing: Gaussian blur sigma value for image smoothing (0 = no smoothing, typical range 0.5-3.0)
        
    Returns:
        tuple: (trimesh.Trimesh, float) - Displaced mesh and average brightness (0.0-1.0)
    """
    # Load and process image
    try:
        # Load original image for display (preserve color)
        original_img = Image.open(image_path)
        img_width, img_height = original_img.size
        
        # Convert to grayscale for processing
        img = original_img.convert('L')
        
        # Note: We do NOT resize the image to target dimensions here
        # Resizing would change the image's aspect ratio and potentially its orientation
        # Instead, we preserve the original image dimensions and scale UV coordinates to fit
        # The target_width and target_height are used for calculating scaling, not for resizing
        
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        # Note: img_array.shape = (height, width) = (rows, columns) in numpy convention
        # For a 3024×4032 image: shape = (4032, 3024) where 4032 is rows (vertical) and 3024 is columns (horizontal)
        
        # Store original grayscale array before blur
        original_grayscale_array = img_array.copy()
        
        # Apply Gaussian blur smoothing if requested
        smoothing_sigma = float(smoothing) if smoothing is not None else 0.0
        if smoothing_sigma > 0.0:
            if progress_callback:
                progress_callback(15.0, f"Applying image smoothing (sigma={smoothing_sigma:.2f})...")
            img_array = ndimage.gaussian_filter(img_array, sigma=smoothing_sigma, mode='wrap')
        
        # Calculate average greyscale value (brightness) after processing
        # This represents the average brightness of the processed image (0.0 = black, 1.0 = white)
        average_brightness = float(np.mean(img_array))
        
        # img_width = horizontal dimension (columns) = will map to U (circumference)
        # img_height = vertical dimension (rows) = will map to V (tube height, brim to brim)
        img_aspect = img_width / img_height if img_height > 0 else 1.0
        
        # Prepare images for display (store as PIL Images)
        # Processed image: blurred grayscale
        processed_img = Image.fromarray((img_array * 255.0).astype(np.uint8), mode='L')
        # Original image: color (before any processing)
        original_img_for_display = original_img.copy()
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Validate that mesh is a Trimesh object, not a Scene
    if isinstance(mesh, trimesh.Scene):
        raise ValueError(
            "Expected Trimesh object but received Scene. "
            "Please extract a mesh from the scene first using scene.geometry.values() or scene.dump(concatenate=True)"
        )
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh object but received {type(mesh)}")
    
    # Get mesh vertices (ensure float32 for memory efficiency)
    vertices = mesh.vertices.astype(np.float32)
    
    # Identify side surface vertices (between brims, outer surface only)
    # These are vertices on the outer surface that are NOT on top/bottom edges or brims
    side_vertex_mask = _identify_side_vertices(vertices)
    
    # Only process side surface vertices
    side_vertices = vertices[side_vertex_mask].astype(np.float32)
    
    if len(side_vertices) == 0:
        # No side vertices found, fallback to outer vertices
        side_vertex_mask = _identify_outer_vertices(vertices)
        side_vertices = vertices[side_vertex_mask]
        if len(side_vertices) == 0:
            # Still no vertices, apply to all (last resort)
            side_vertex_mask = np.ones(len(vertices), dtype=bool)
            side_vertices = vertices
    
    # Calculate UV coordinates for side vertices only
    # Map 3D positions to 2D UV coordinates using cylindrical mapping
    uv_coords = uv_computation(side_vertices).astype(np.float32)
    
    # Calculate surface dimensions (excluding brims) for scaling
    z = side_vertices[:, 2]
    z_min = np.min(z)
    z_max = np.max(z)
    surface_height = z_max - z_min
    
    # Clean up intermediate if not needed later
    del z
    
    # Map image to surface while preserving original orientation and aspect ratio
    # CRITICAL: Image dimensions map as follows:
    #   - Image HEIGHT (rows, vertical dimension) → V coordinate (tube vertical, brim to brim)
    #   - Image WIDTH (columns, horizontal dimension) → U coordinate (tube circumference, wraps around)
    # This ensures portrait images stay upright and landscape images maintain their orientation
    #
    # We do NOT scale the UV coordinates - we let the image map naturally to the surface
    # The image will wrap around the circumference (U) and map to the full height (V)
    # This preserves the image's original orientation completely
    #
    # Note: The image may wrap multiple times around the circumference or only partially cover it,
    # depending on the aspect ratios. This is intentional to preserve orientation.
    
    # Ensure V coordinates stay within [0, 1] to prevent overlapping brims
    # U coordinates can wrap (handled in _sample_displacement)
    uv_coords[:, 1] = np.clip(uv_coords[:, 1], 0.0, 1.0)
    
    # Sample displacement values from image
    # Convert intensity from cm to mm to match mesh units
    intensity_mm = intensity * 10.0
    offset_mm = offset * 10.0
    
    # Sample displacement values from image
    num_side_vertices = len(side_vertices)
    raw_samples = np.zeros(num_side_vertices, dtype=np.float32)
    
    if progress_callback:
        progress_callback(20.0, f"Sampling displacement from image ({num_side_vertices:,} vertices)...")
    
    # Process all UV coordinates at once
    raw_samples = bilinear_interpolation(
        img_array,
        uv_coords,
        mode='wrap'
    ).astype(np.float32)
    
    if progress_callback:
        progress_callback(35.0, f"Sampled displacement for {num_side_vertices:,} vertices")
    
    # Clean up intermediate
    del uv_coords
    gc.collect()
    
    # Thickness-aware normalized mapping:
    # allowed_inward = thickness - min_clearance
    # intensity in [0,1] scales the allowed_inward; brightest pixels map to allowed_inward * intensity
    # If thickness/min_clearance unavailable, fall back to old behavior using intensity_mm/offset_mm.
    side_displacement = None
    allowed_inward = None
    if thickness_mm is not None:
        try:
            tmm = float(thickness_mm)
            cmm = float(min_clearance_mm if min_clearance_mm is not None else 0.0)
            allowed_inward = max(0.0, tmm - cmm)
        except Exception:
            allowed_inward = None

    if allowed_inward is not None:
        # Normalize intensity to [0,1]
        try:
            intensity_norm = float(intensity)
        except Exception:
            intensity_norm = 1.0
        intensity_norm = max(0.0, min(1.0, intensity_norm))
        max_inward = allowed_inward * intensity_norm
        # Map raw samples (0..1) to 0..max_inward and add offset in mm, then clamp
        side_displacement = raw_samples * max_inward + offset_mm
        if max_inward <= 0.0:
            side_displacement.fill(0.0)
        else:
            np.clip(side_displacement, 0.0, max_inward, out=side_displacement)
    else:
        # Fallback: legacy scaling with physical intensity in mm and offset
        side_displacement = raw_samples * intensity_mm + offset_mm
        # No specific ceiling; keep non-negative inward displacement
        np.clip(side_displacement, 0.0, None, out=side_displacement)

    # Create displacement array for all vertices (zero for non-side vertices)
    displacement = np.zeros(len(vertices), dtype=np.float32)
    displacement[side_vertex_mask] = side_displacement.astype(np.float32)
    
    # Clean up intermediate
    del side_displacement
    gc.collect()
    
    # Apply displacement vertically (along Z-axis) to create height variations
    # This creates bumps/indentations that extend outward/inward from the surface
    # Displacement is radial (along surface normal direction, but only on side surface)
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        normals = mesh.vertex_normals.astype(np.float32)
    else:
        # Create temporary mesh for normal computation (will be cleaned up)
        mesh_temp = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        mesh_temp.fix_normals()
        normals = mesh_temp.vertex_normals.astype(np.float32)
        del mesh_temp
        gc.collect()
    
    # Apply displacement along normals (radial displacement)
    # Only affects side surface vertices
    # Negative displacement for inward effect (subtract instead of add)
    displaced_vertices = vertices.copy().astype(np.float32)
    side_vertex_count = np.sum(side_vertex_mask)
    side_vertex_indices = np.where(side_vertex_mask)[0]
    
    if progress_callback:
        progress_callback(50.0, f"Applying displacement to {side_vertex_count:,} vertices...")
    
    # CPU vectorized displacement (process all at once)
    displacement_vectors = (normals * displacement[:, np.newaxis]).astype(np.float32)
    displaced_vertices[side_vertex_indices] -= displacement_vectors[side_vertex_indices].astype(np.float32)
    
    # Clean up intermediates
    del normals, displacement, displacement_vectors
    gc.collect()
    
    # Update progress
    if progress_callback:
        progress_callback(60.0, f"Displaced {side_vertex_count:,} vertices")
    
    # Create new mesh with displaced vertices
    # Ensure faces are uint32 for memory efficiency
    faces_uint32 = mesh.faces.astype(np.uint32) if mesh.faces.dtype != np.uint32 else mesh.faces
    displaced_mesh = trimesh.Trimesh(
        vertices=displaced_vertices,
        faces=faces_uint32,
        process=False  # Avoid expensive processing
    )
    
    # Recompute normals after displacement
    displaced_mesh.fix_normals()
    
    # Clean up intermediates
    del displaced_vertices, faces_uint32
    gc.collect()
    
    # Convert images to base64 for JSON serialization
    def image_to_base64(pil_img):
        """Convert PIL Image to base64 data URL"""
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_b64}"
    
    processed_img_b64 = image_to_base64(processed_img)
    original_img_b64 = image_to_base64(original_img_for_display)
    
    return displaced_mesh, average_brightness, processed_img_b64, original_img_b64, img_width, img_height


def _identify_side_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Identify side surface vertices (between brims, outer surface only).
    Excludes top/bottom edges, brims, and inner surface.
    Only includes vertices on the actual tube surface grid, not brim extensions.
    
    Args:
        vertices: Vertex positions
        
    Returns:
        np.ndarray: Boolean mask indicating side surface vertices
    """
    # Calculate radius (distance from z-axis) for each vertex
    radius = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    z = vertices[:, 2]
    
    # Find height range
    z_min = np.min(z)
    z_max = np.max(z)
    z_range = z_max - z_min
    
    # Step 1: Identify outer surface vertices (exclude inner surface)
    outer_mask = _identify_outer_vertices(vertices)
    
    # Step 2: Identify the typical tube radius (not including brims which extend outward)
    # Use the median radius of outer vertices in the middle 60% of the height
    # This avoids brim areas at top and bottom
    middle_start = z_min + z_range * 0.2
    middle_end = z_max - z_range * 0.2
    middle_mask = (z >= middle_start) & (z <= middle_end) & outer_mask
    
    if np.sum(middle_mask) > 0:
        middle_radii = radius[middle_mask]
        tube_radius_median = np.median(middle_radii)
        tube_radius_std = np.std(middle_radii) if len(middle_radii) > 1 else tube_radius_median * 0.1
        # Allow tolerance for radius variation (tapered tubes) - use 2 standard deviations
        tube_radius_max = tube_radius_median + max(tube_radius_std * 2, tube_radius_median * 0.1)
    else:
        # Fallback: use median of all outer vertices
        all_outer_radii = radius[outer_mask]
        if len(all_outer_radii) > 0:
            tube_radius_median = np.median(all_outer_radii)
            tube_radius_max = tube_radius_median * 1.15  # 15% tolerance
        else:
            # Last resort fallback
            tube_radius_median = np.median(radius) if len(radius) > 0 else 10.0
            tube_radius_max = tube_radius_median * 1.15
    
    # Step 3: Exclude brim vertices (they have larger radius than the tube)
    # Brims extend outward, so their radius is significantly larger than tube radius
    # Use a very strict threshold to exclude even the sloped connection areas
    # Only allow vertices that are very close to the tube radius (within 1% tolerance)
    # This excludes slope vertices which may be slightly above tube radius
    brim_threshold = tube_radius_median * 1.01  # Only 1% tolerance above median tube radius
    not_brim_mask = radius <= brim_threshold
    
    # Step 4: Identify the actual tube surface z-range (excluding brim areas)
    # Find z-coordinates where vertices are at tube radius (not brim radius)
    # Group by z-level and find where radius matches tube radius
    z_rounded = np.round(z, decimals=1)
    z_unique = np.unique(z_rounded)
    
    tube_z_levels = []
    for z_level in z_unique:
        level_mask = (z_rounded == z_level) & outer_mask
        if np.sum(level_mask) > 0:
            level_radii = radius[level_mask]
            # Check if this level is primarily tube surface (not brim)
            # Use a threshold that allows for some variation but excludes primarily brim levels
            level_brim_threshold = tube_radius_median * 1.01
            # Count how many vertices are at tube radius (within 1% of median)
            tube_radius_count = np.sum((level_radii <= level_brim_threshold) & (level_radii >= tube_radius_median * 0.99))
            total_count = len(level_radii)
            
            # Include if most vertices are at tube radius (70% threshold - more lenient to avoid gaps)
            # This allows levels that are mostly tube surface, even if they have some slope vertices at edges
            if total_count > 0 and (tube_radius_count / total_count > 0.7):
                level_median_radius = np.median(level_radii)
                # Also check that median is reasonable (not clearly a brim level)
                if level_median_radius <= tube_radius_median * 1.02:  # Allow 2% tolerance for median
                    tube_z_levels.append(z_level)
    
    if len(tube_z_levels) > 0:
        # Find the exact z-positions where the tube surface ends (where brim slopes begin)
        # tube_z_levels contains levels that are primarily tube surface (>=70% tube vertices)
        # The min and max of these levels define the exact boundaries of the tube surface grid
        tube_z_min = np.min(tube_z_levels)
        tube_z_max = np.max(tube_z_levels)
        
        # Exclude the entire first and last rows of the tube surface grid
        # This ensures the brim connection is perfectly smooth without any displacement artifacts
        # Find vertices at the boundary z-levels (the exact first and last rows)
        boundary_z_tolerance = 0.2  # 0.2mm tolerance for boundary detection (very tight)
        at_bottom_boundary = np.abs(z - tube_z_min) < boundary_z_tolerance
        at_top_boundary = np.abs(z - tube_z_max) < boundary_z_tolerance
        at_boundary = at_bottom_boundary | at_top_boundary
        
        # Exclude ALL vertices at the boundary z-levels to ensure smooth brim connection
        # This includes both the tube surface vertices and any slope vertices at these levels
        # Use the exact boundaries - exclude all boundary vertices for smooth brim connection
        # The tube surface ends exactly at these z-positions where the brim slopes begin
        # Use a small tolerance only for floating point precision
        z_tolerance = 0.1  # 0.1mm tolerance for floating point precision
        tube_z_mask = (z >= tube_z_min - z_tolerance) & (z <= tube_z_max + z_tolerance) & ~at_boundary
    else:
        # Fallback: exclude top and bottom 20% more aggressively (to exclude slopes)
        z_tolerance = z_range * 0.20
        tube_z_min = z_min + z_tolerance
        tube_z_max = z_max - z_tolerance
        tube_z_mask = (z >= tube_z_min) & (z <= tube_z_max)
    
    # Additional check: exclude vertices that are clearly at brim radius or on slopes
    # This catches vertices on the sloped connection that might have passed other checks
    # Use a very strict threshold - only allow vertices within 1% of tube radius
    strict_brim_threshold = tube_radius_median * 1.01  # Only 1% above median tube radius
    strict_not_brim_mask = radius <= strict_brim_threshold
    
    # Step 5: Combine all filters
    # Side vertices = outer surface + tube radius + within tube z-range + not inner
    # Use tube_radius_median for inner surface exclusion
    # Apply strict brim exclusion to catch any remaining brim vertices
    side_mask = (
        outer_mask & 
        not_brim_mask & 
        strict_not_brim_mask &  # Additional strict brim check
        tube_z_mask &
        (radius >= tube_radius_median * 0.7)  # Exclude inner surface (smaller radius)
    )
    
    return side_mask


def _identify_outer_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Identify outer surface vertices by finding vertices with maximum radius
    at each height level. For a hollow tube, outer vertices have larger radius.
    Uses GPU acceleration if available.
    
    Args:
        vertices: Vertex positions
        
    Returns:
        np.ndarray: Boolean mask indicating outer vertices
    """
    # Use CPU-based implementation
    return identify_outer_vertices(vertices)


def _compute_cylindrical_uv(vertices: np.ndarray) -> np.ndarray:
    """
    Compute UV coordinates using cylindrical mapping
    Maps the side surface of a cylinder to a 2D image
    
    Args:
        vertices: Vertex positions (side surface vertices only)
        
    Returns:
        np.ndarray: UV coordinates in range [0, 1]
    """
    # Calculate cylindrical coordinates
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # U coordinate: angle around the cylinder (0 to 1)
    # This wraps around the cylinder horizontally
    angle = np.arctan2(y, x)
    u = (angle + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # V coordinate: height along the cylinder (0 to 1)
    # This maps from bottom to top of the side surface
    # Note: vertices should already be filtered to side surface only
    z_min = np.min(z)
    z_max = np.max(z)
    z_range = z_max - z_min
    if z_range > 0.001:  # Avoid division by zero
        v = (z - z_min) / z_range
        # Ensure V is in [0, 1] range
        v = np.clip(v, 0.0, 1.0)
        # Flip over x-axis: invert V coordinate so image appears right-side up
        v = 1.0 - v
    else:
        # All vertices at same height (shouldn't happen for side surface)
        v = np.ones_like(z)  # Flipped: use 1.0 instead of 0.0
    
    return np.column_stack([u, v])


def _sample_displacement(
    image: np.ndarray,
    uv_coords: np.ndarray,
    intensity: float = 1.0,
    offset: float = 0.0
) -> np.ndarray:
    """
    Sample displacement values from image at UV coordinates
    
    Args:
        image: Grayscale image array (normalized to [0, 1])
        uv_coords: UV coordinates for sampling
        intensity: Displacement intensity multiplier
        offset: Base offset for displacement
        
    Returns:
        np.ndarray: Displacement values for each vertex
    """
    # image.shape = (height, width) = (rows, columns) in numpy convention
    height, width = image.shape
    # height = number of rows (vertical dimension) = image height
    # width = number of columns (horizontal dimension) = image width
    
    # Convert UV coordinates to pixel coordinates
    # U coordinate (circumference, wraps around) → image width (columns, horizontal)
    # V coordinate (tube vertical, brim to brim) → image height (rows, vertical)
    u = (uv_coords[:, 0] % 1.0) * width   # Map U to columns (horizontal)
    v = (uv_coords[:, 1] % 1.0) * height   # Map V to rows (vertical)
    
    # Clamp coordinates to valid range
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    
    # Sample using bilinear interpolation
    # map_coordinates uses [row, column] = [v, u] format
    # This correctly maps: V (vertical) → rows, U (circumference) → columns
    displacement = ndimage.map_coordinates(
        image,
        [v, u],  # [rows, columns] = [vertical, horizontal]
        order=1,
        mode='wrap'
    )
    
    # Apply intensity and offset
    displacement = displacement * intensity + offset
    
    return displacement


def apply_brims_parametric(
    mesh: trimesh.Trimesh,
    tube_height: float,
    tube_radius: float,
    brim_radius: float,
    brim_height: float,
    slope_height: float,
    top_brim: bool = True,
    bottom_brim: bool = True,
    progress_callback=None
) -> trimesh.Trimesh:
    """
    Procedurally displace outer surface vertices to form top and bottom brims.
    - Brim radius: target outer radius at the very ends (y in [0, brim_height] and [H-brim_height, H])
    - Slope height: vertical ramp distance to blend from tube_radius to brim_radius
    - top_brim: If True, apply top brim (default: True)
    - bottom_brim: If True, apply bottom brim (default: True)
    
    Only outer vertices are displaced; inner surface remains unchanged.
    """
    if isinstance(mesh, trimesh.Scene):
        raise ValueError("Expected Trimesh for brim displacement")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh, got {type(mesh)}")

    vertices = mesh.vertices.astype(np.float32)
    if len(vertices) == 0:
        return mesh

    # Identify outer vertices
    outer_mask = _identify_outer_vertices(vertices)
    outer_idx = np.nonzero(outer_mask)[0]
    if progress_callback:
        progress_callback(82.0, f"Identified {len(outer_idx)} outer vertices")

    # Geometry parameters
    H = float(max(0.0, tube_height))
    r_tube = float(max(1e-6, tube_radius))
    r_brim = float(max(r_tube, brim_radius))
    h_brim = float(max(0.0, brim_height))
    h_slope = float(max(0.0, slope_height))

    # Precompute deltas
    delta_r = r_brim - r_tube
    # Check if any brims are enabled and if parameters are valid
    if not top_brim and not bottom_brim:
        # No brims enabled, nothing to do
        if progress_callback:
            progress_callback(90.0, "No brims enabled")
        return mesh
    if delta_r <= 1e-9 or h_brim <= 1e-9:
        # Nothing to do
        if progress_callback:
            progress_callback(90.0, "Brim parameters indicate no displacement needed")
        return mesh

    # Z is up
    z = vertices[:, 2]
    x = vertices[:, 0]
    y = vertices[:, 1]
    r_current = np.sqrt(x * x + y * y)

    # Build target radius function along Y for bottom and top zones
    # Bottom zone: z in [0, h_brim]
    # Top zone: z in [H - h_brim, H]
    target_r = np.full_like(r_current, r_tube, dtype=np.float32)

    # Bottom brim zone: flat brim of height h_brim at the end (z in [0, h_brim]),
    # then slope towards the tube in (h_brim, h_brim + h_slope]
    if bottom_brim and h_brim > 1e-9:
        # Flat brim portion [0, h_brim]
        bottom_flat_mask = (z >= 0.0) & (z <= h_brim)
        target_r[bottom_flat_mask] = r_brim
        # Slope portion (h_brim, h_brim + h_slope]
        if h_slope > 1e-9:
            bottom_slope_mask = (z > h_brim) & (z <= h_brim + h_slope)
            # s = 0 at tube side (z = h_brim + h_slope), s = 1 at start of slope (z = h_brim)
            s = np.zeros_like(z, dtype=np.float32)
            s[bottom_slope_mask] = (h_brim + h_slope - z[bottom_slope_mask]) / h_slope
            s_smooth = s * s * (3.0 - 2.0 * s)
            target_r[bottom_slope_mask] = r_tube + delta_r * s_smooth[bottom_slope_mask]

    # Top: mirror
    if top_brim and H > 0.0:
        # Top brim: slope first [H - h_brim - h_slope, H - h_brim), then flat [H - h_brim, H]
        if h_brim > 1e-9:
            # Flat brim portion [H - h_brim, H]
            top_flat_start = H - h_brim
            top_flat_mask = (z >= top_flat_start) & (z <= H)
            target_r[top_flat_mask] = np.maximum(target_r[top_flat_mask], r_brim)
            # Slope portion [H - h_brim - h_slope, H - h_brim)
            if h_slope > 1e-9:
                top_slope_start = H - h_brim - h_slope
                top_slope_end = H - h_brim
                top_slope_mask = (z >= top_slope_start) & (z < top_slope_end)
                # s = 0 at tube side (z = top_slope_start), s = 1 at start of flat (z = top_slope_end)
                s = np.zeros_like(z, dtype=np.float32)
                s[top_slope_mask] = (z[top_slope_mask] - top_slope_start) / h_slope
                s_smooth = s * s * (3.0 - 2.0 * s)
                target_r[top_slope_mask] = np.maximum(target_r[top_slope_mask], r_tube + delta_r * s_smooth[top_slope_mask])

    # Only apply to outer vertices; inner remain unchanged
    # Compute needed radial outward move for outer vertices to reach target_r
    desired_r = target_r
    move_outward = np.clip(desired_r - r_current, 0.0, None)

    if progress_callback:
        progress_callback(88.0, "Applying radial displacement for brims...")

    # Apply displacement: move along radial direction in XY plane (Z-up)
    # Avoid division by zero by skipping tiny radius vertices
    denom = np.maximum(r_current, 1e-9)
    scale = move_outward / denom

    # Only affect outer vertices
    scale[~outer_mask] = 0.0

    vertices[:, 0] = x + x * scale
    vertices[:, 1] = y + y * scale
    
    # Clean up intermediates
    del x, y, r_current, scale, target_r, move_outward
    gc.collect()

    # Ensure faces are uint32 for memory efficiency
    faces_uint32 = mesh.faces.astype(np.uint32) if mesh.faces.dtype != np.uint32 else mesh.faces
    out_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces_uint32, process=False)
    
    # Clean up
    del vertices, faces_uint32
    gc.collect()
    out_mesh.fix_normals()
    if progress_callback:
        progress_callback(92.0, "Brim displacement complete")
    return out_mesh
