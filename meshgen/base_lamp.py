import numpy as np
import trimesh
import gc
from .displace import apply_brims_parametric


def _create_parametric_tube(height, radius, mm_per_pixel, thickness, progress_callback=None):
    """
    Create a straight hollow tube (no brims) using mm_per_pixel as target grid size.
    Returns vertices (N,3) and faces (M,3).
    """
    if progress_callback:
        progress_callback(5.0, "Initializing tube generation...")

    segment_size = max(1e-6, float(mm_per_pixel))

    # Resolution
    circumference = 2.0 * np.pi * radius
    num_circ = max(8, int(np.ceil(circumference / segment_size)))
    num_h = max(2, int(np.ceil(height / segment_size)))

    if progress_callback:
        progress_callback(10.0, f"Resolution: {num_circ} circular x {num_h} height")

    r_outer = float(radius)
    r_inner = max(1e-6, r_outer - float(thickness))

    # Rings along Z (0..height), both inner and outer share Z (Z is up)
    z_profile = np.linspace(0.0, float(height), num_h + 1, dtype=np.float32)
    num_rings_outer = len(z_profile)
    # Inside does not need ring segmentation: only bottom and top rings
    z_profile_inner = np.array([0.0, float(height)], dtype=np.float32)
    num_rings_inner = len(z_profile_inner)

    theta = np.linspace(0.0, 2.0 * np.pi, num_circ, endpoint=False, dtype=np.float32)
    x_unit = np.cos(theta).astype(np.float32)
    y_unit = np.sin(theta).astype(np.float32)

    if progress_callback:
        progress_callback(20.0, "Generating vertices...")

    # Outer (Z-up), circle in XY (ensure float32)
    outer_x = (r_outer * x_unit[None, :].repeat(num_rings_outer, axis=0)).astype(np.float32)
    outer_y = (r_outer * y_unit[None, :].repeat(num_rings_outer, axis=0)).astype(np.float32)
    outer_z_up = z_profile[:, None].repeat(num_circ, axis=1).astype(np.float32)

    # Inner (Z-up), circle in XY (ensure float32)
    inner_x = (r_inner * x_unit[None, :].repeat(num_rings_inner, axis=0)).astype(np.float32)
    inner_y = (r_inner * y_unit[None, :].repeat(num_rings_inner, axis=0)).astype(np.float32)
    inner_z_up = z_profile_inner[:, None].repeat(num_circ, axis=1).astype(np.float32)

    inner_verts = np.stack([inner_x.ravel(), inner_y.ravel(), inner_z_up.ravel()], axis=1, dtype=np.float32)
    outer_verts = np.stack([outer_x.ravel(), outer_y.ravel(), outer_z_up.ravel()], axis=1, dtype=np.float32)
    
    # Clean up intermediates
    del inner_x, inner_y, inner_z_up, outer_x, outer_y, outer_z_up
    gc.collect()
    
    vertices = np.vstack([inner_verts, outer_verts]).astype(np.float32)
    
    # Clean up intermediates
    del inner_verts, outer_verts
    gc.collect()

    if progress_callback:
        progress_callback(30.0, f"Vertices: {len(vertices)}")

    # Side faces
    faces = []
    outer_offset = num_rings_inner * num_circ
    quads_total = (num_rings_outer - 1) * num_circ

    if progress_callback:
        progress_callback(40.0, "Generating wall faces...")

    # Inner wall faces (only between bottom and top inner rings)
    for i in range(num_circ):
        i_next = (i + 1) % num_circ
        v1_in = 0 * num_circ + i
        v2_in = 0 * num_circ + i_next
        v3_in = 1 * num_circ + i_next
        v4_in = 1 * num_circ + i
        faces.append([v1_in, v2_in, v4_in])
        faces.append([v2_in, v3_in, v4_in])

    # Outer wall faces (full segmentation along height)
    for j in range(num_rings_outer - 1):
        for i in range(num_circ):
            i_next = (i + 1) % num_circ

            # Outer quad (CCW, outward)
            v1_out = outer_offset + j * num_circ + i
            v2_out = outer_offset + j * num_circ + i_next
            v3_out = outer_offset + (j + 1) * num_circ + i_next
            v4_out = outer_offset + (j + 1) * num_circ + i
            faces.append([v1_out, v4_out, v2_out])
            faces.append([v2_out, v4_out, v3_out])

    if progress_callback:
        progress_callback(60.0, "Generating caps...")

    # Bottom cap (z=0): connect inner and outer ring 0
    j_bottom = 0
    for i in range(num_circ):
        i_next = (i + 1) % num_circ
        v1_in = 0 * num_circ + i  # inner bottom ring
        v2_in = 0 * num_circ + i_next
        v1_out = outer_offset + j_bottom * num_circ + i
        v2_out = outer_offset + j_bottom * num_circ + i_next
        faces.append([v1_in, v2_out, v2_in])
        faces.append([v1_in, v1_out, v2_out])

    # Top cap (z=height): connect inner and outer ring last
    j_top = num_rings_outer - 1
    for i in range(num_circ):
        i_next = (i + 1) % num_circ
        v1_in = 1 * num_circ + i  # inner top ring
        v2_in = 1 * num_circ + i_next
        v1_out = outer_offset + j_top * num_circ + i
        v2_out = outer_offset + j_top * num_circ + i_next
        faces.append([v1_in, v2_in, v2_out])
        faces.append([v1_in, v2_out, v1_out])

    # Use uint32 for face indices (supports up to 4.2B vertices, more memory efficient than int32)
    faces = np.array(faces, dtype=np.uint32)

    if progress_callback:
        progress_callback(70.0, f"Faces: {len(faces)}")

    return vertices, faces


def create_base_lamp(
    height,
    mm_per_pixel,
    thickness,
    brim_height,
    diameter,  # diameter in mm
    brim_diameter,  # brim diameter in mm
    brim_slope_height=None,
    brim_overhang_angle=None,
    top_brim=True,
    bottom_brim=True,
    progress_callback=None
):
    """
    Generates a straight tube at specified resolution, then procedurally displaces
    the outer surface to form top and bottom brims with a controllable slope.
    """
    if progress_callback:
        progress_callback(2.0, "Starting base lamp generation (tube + brim displacement)...")

    # Inputs from UI/API:
    # - height, diameter, brim_diameter, brim_height are in mm
    # - thickness is in mm
    # Internally we operate in mm to match mm_per_pixel grid resolution
    height_mm = float(height)
    
    # Convert diameter to radius for internal calculations
    if diameter is None:
        raise ValueError("diameter must be provided")
    radius_mm = float(diameter) / 2.0
    
    # Convert brim_diameter to brim_radius for internal calculations
    if brim_diameter is None:
        raise ValueError("brim_diameter must be provided")
    brim_radius_mm = float(brim_diameter) / 2.0
    
    brim_height_mm = float(brim_height)
    thickness_mm = float(thickness)  # already in mm

    vertices, faces = _create_parametric_tube(
        height=height_mm,
        radius=radius_mm,
        mm_per_pixel=mm_per_pixel,
        thickness=thickness_mm,
        progress_callback=progress_callback
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Clean up intermediates (vertices and faces are now in mesh)
    del vertices, faces
    gc.collect()

    # Determine effective brim radius (supports extension or absolute)
    radius_mm_val = float(radius_mm)
    effective_brim_radius_mm = radius_mm_val + brim_radius_mm if brim_radius_mm <= radius_mm_val else brim_radius_mm

    # Compute slope height preference (dynamic w.r.t. brim size):
    # 1) If angle provided, slope_height = (effective_brim_radius - radius) * tan(angle_from_horizontal)
    # 2) Else if brim_slope_height provided, use it
    # 3) Else default to brim_height
    delta_r_mm = max(0.0, float(effective_brim_radius_mm) - float(radius_mm))
    slope_h = None
    if brim_overhang_angle is not None:
        try:
            angle_rad = np.deg2rad(float(brim_overhang_angle))
            tan_val = np.tan(angle_rad)
            slope_h = delta_r_mm * tan_val
        except Exception:
            slope_h = None
    if slope_h is None:
        slope_h = float(brim_height_mm) if brim_slope_height is None else float(brim_slope_height)
    # Slope is outside the flat brim; ensure non-negative and finite
    slope_h = 0.0 if not np.isfinite(slope_h) else float(max(0.0, slope_h))

    if progress_callback:
        progress_callback(80.0, "Applying brim displacement...")

    displaced_mesh = apply_brims_parametric(
        mesh=mesh,
        tube_height=float(height_mm),
        tube_radius=float(radius_mm),
        brim_radius=effective_brim_radius_mm,
        brim_height=float(brim_height_mm),
        slope_height=float(slope_h),
        top_brim=bool(top_brim),
        bottom_brim=bool(bottom_brim),
        progress_callback=progress_callback
    )

    displaced_mesh.fix_normals()
    
    # Clean up original mesh if not needed
    del mesh
    gc.collect()

    if progress_callback:
        progress_callback(95.0, "Finalizing mesh data...")

    # Ensure output arrays are memory-efficient types
    return {
        'vertices': displaced_mesh.vertices.astype(np.float32).copy(),
        'faces': displaced_mesh.faces.astype(np.uint32).copy()
    }