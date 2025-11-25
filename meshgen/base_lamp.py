import numpy as np
import trimesh
import gc
from .displace import apply_brims_parametric


def _create_spoke(start_pos, end_pos, thickness, height, mm_per_pixel, progress_callback=None):
    """
    Create a rectangular spoke (rectangular prism) from start_pos to end_pos.
    start_pos and end_pos are 3D coordinates (x, y, z) in mm.
    thickness is the width of the spoke in mm (perpendicular to direction).
    height is the vertical height of the spoke in mm.
    Returns vertices (N,3) and faces (M,3).
    """
    if progress_callback:
        progress_callback(0.0, "Generating rectangular spoke...")
    
    segment_size = max(1e-6, float(mm_per_pixel))
    
    # Calculate spoke direction and length
    direction = np.array(end_pos) - np.array(start_pos)
    length = np.linalg.norm(direction)
    if length < 1e-6:
        # Degenerate spoke, return empty mesh
        return np.array([], dtype=np.float32).reshape(0, 3), np.array([], dtype=np.int32).reshape(0, 3)
    
    direction_normalized = direction / length
    
    # Resolution along length
    num_length = max(2, int(np.ceil(length / segment_size)))
    # Resolution along height
    num_height = max(2, int(np.ceil(height / segment_size)))
    
    # Create local coordinate system
    # For horizontal spokes, we want:
    # - u: perpendicular to direction in XY plane (for thickness)
    # - v: Z direction (for vertical height)
    
    # Get a vector perpendicular to direction in XY plane
    # If direction is mostly horizontal, use Z as reference
    if abs(direction_normalized[2]) < 0.9:
        # Direction is mostly horizontal, use Z as reference
        perp = np.array([0, 0, 1])
        u = np.cross(direction_normalized, perp)
        u = u / np.linalg.norm(u)
        # v is the Z direction
        v = np.array([0, 0, 1])
    else:
        # Direction is mostly vertical, use X as reference
        perp = np.array([1, 0, 0])
        u = np.cross(direction_normalized, perp)
        u = u / np.linalg.norm(u)
        # v is perpendicular to both
        v = np.cross(direction_normalized, u)
        v = v / np.linalg.norm(v)
        # Ensure v points in positive Z direction
        if v[2] < 0:
            v = -v
    
    # For rectangular cross-section
    # Create a rectangle: thickness/2 in +u and -u directions
    # Height extends from z=0 (bottom) to z=height (top)
    half_thickness = float(thickness) / 2.0
    
    # Generate vertices
    # Segment along length and height
    t_length_values = np.linspace(0.0, 1.0, num_length + 1)
    t_height_values = np.linspace(0.0, 1.0, num_height + 1)
    vertices = []
    
    for t_len in t_length_values:
        pos = np.array(start_pos) + t_len * direction
        for t_h in t_height_values:
            # Position along height (from 0 to height)
            height_offset = t_h * height * v
            # Position along thickness (from -half_thickness to +half_thickness)
            for side in [-1, 1]:
                thickness_offset = side * half_thickness * u
                vertex = pos + height_offset + thickness_offset
                vertices.append(vertex)
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Generate faces
    faces = []
    verts_per_cross_section = (num_height + 1) * 2  # 2 sides (left/right) per height level
    
    # Helper function to get vertex index
    def get_vertex_idx(j, h, side):
        """Get vertex index for cross-section j, height h, side (0=left, 1=right)"""
        return j * verts_per_cross_section + h * 2 + side
    
    # Side faces (left and right, along the length)
    for j in range(num_length):
        for h in range(num_height):
            # Left side face
            v1 = get_vertex_idx(j, h, 0)      # bottom-left
            v2 = get_vertex_idx(j, h+1, 0)    # top-left
            v3 = get_vertex_idx(j+1, h+1, 0)  # top-left, next section
            v4 = get_vertex_idx(j+1, h, 0)    # bottom-left, next section
            faces.append([v1, v2, v4])
            faces.append([v2, v3, v4])
            
            # Right side face
            v1 = get_vertex_idx(j, h, 1)      # bottom-right
            v2 = get_vertex_idx(j, h+1, 1)    # top-right
            v3 = get_vertex_idx(j+1, h+1, 1)  # top-right, next section
            v4 = get_vertex_idx(j+1, h, 1)    # bottom-right, next section
            faces.append([v1, v4, v2])
            faces.append([v2, v4, v3])
    
    # Top and bottom faces (along the length)
    for j in range(num_length):
        # Bottom face (h=0)
        v1 = get_vertex_idx(j, 0, 0)      # left
        v2 = get_vertex_idx(j, 0, 1)      # right
        v3 = get_vertex_idx(j+1, 0, 1)    # right, next section
        v4 = get_vertex_idx(j+1, 0, 0)    # left, next section
        faces.append([v1, v2, v4])
        faces.append([v2, v3, v4])
        
        # Top face (h=num_height)
        v1 = get_vertex_idx(j, num_height, 0)      # left
        v2 = get_vertex_idx(j, num_height, 1)      # right
        v3 = get_vertex_idx(j+1, num_height, 1)    # right, next section
        v4 = get_vertex_idx(j+1, num_height, 0)    # left, next section
        faces.append([v1, v4, v2])
        faces.append([v2, v4, v3])
    
    # Front and back faces (end caps)
    # Start cap (at start_pos, j=0)
    for h in range(num_height):
        v1 = get_vertex_idx(0, h, 0)      # left
        v2 = get_vertex_idx(0, h, 1)      # right
        v3 = get_vertex_idx(0, h+1, 1)    # right, next height
        v4 = get_vertex_idx(0, h+1, 0)    # left, next height
        faces.append([v1, v2, v4])
        faces.append([v2, v3, v4])
    
    # End cap (at end_pos, j=num_length)
    for h in range(num_height):
        v1 = get_vertex_idx(num_length, h, 0)      # left
        v2 = get_vertex_idx(num_length, h, 1)      # right
        v3 = get_vertex_idx(num_length, h+1, 1)    # right, next height
        v4 = get_vertex_idx(num_length, h+1, 0)    # left, next height
        faces.append([v1, v4, v2])
        faces.append([v2, v4, v3])
    
    # Use uint32 for face indices
    faces = np.array(faces, dtype=np.uint32)
    
    return vertices, faces


def _create_hub_tube(center_pos, height, radius, mm_per_pixel, thickness, progress_callback=None):
    """
    Create a hollow tube (hub) at center_pos.
    center_pos is the center of the bottom of the tube (x, y, z) in mm.
    height, radius, thickness are in mm.
    Returns vertices (N,3) and faces (M,3).
    """
    if progress_callback:
        progress_callback(0.0, "Generating hub tube...")
    
    segment_size = max(1e-6, float(mm_per_pixel))
    
    # Resolution
    circumference = 2.0 * np.pi * radius
    num_circ = max(8, int(np.ceil(circumference / segment_size)))
    num_h = max(2, int(np.ceil(height / segment_size)))
    
    r_outer = float(radius)
    r_inner = max(1e-6, r_outer - float(thickness))
    
    # Generate circular profile (use float32)
    theta = np.linspace(0.0, 2.0 * np.pi, num_circ, endpoint=False, dtype=np.float32)
    x_unit = np.cos(theta).astype(np.float32)
    y_unit = np.sin(theta).astype(np.float32)
    
    # Z profile (use float32 for memory efficiency)
    z_profile = np.linspace(0.0, float(height), num_h + 1, dtype=np.float32)
    num_rings_outer = len(z_profile)
    z_profile_inner = np.array([0.0, float(height)], dtype=np.float32)
    num_rings_inner = len(z_profile_inner)
    
    # Generate vertices
    center = np.array(center_pos)
    
    # Inner vertices
    inner_verts = []
    for z in z_profile_inner:
        for i in range(num_circ):
            x = r_inner * x_unit[i]
            y = r_inner * y_unit[i]
            inner_verts.append(center + np.array([x, y, z]))
    
    # Outer vertices
    outer_verts = []
    for z in z_profile:
        for i in range(num_circ):
            x = r_outer * x_unit[i]
            y = r_outer * y_unit[i]
            outer_verts.append(center + np.array([x, y, z]))
    
    inner_verts = np.array(inner_verts, dtype=np.float32)
    outer_verts = np.array(outer_verts, dtype=np.float32)
    vertices = np.vstack([inner_verts, outer_verts]).astype(np.float32)
    
    # Clean up intermediates
    del inner_verts, outer_verts
    gc.collect()
    
    # Generate faces (similar to _create_parametric_tube)
    faces = []
    outer_offset = num_rings_inner * num_circ
    
    # Inner wall faces
    for i in range(num_circ):
        i_next = (i + 1) % num_circ
        v1_in = 0 * num_circ + i
        v2_in = 0 * num_circ + i_next
        v3_in = 1 * num_circ + i_next
        v4_in = 1 * num_circ + i
        faces.append([v1_in, v2_in, v4_in])
        faces.append([v2_in, v3_in, v4_in])
    
    # Outer wall faces
    for j in range(num_rings_outer - 1):
        for i in range(num_circ):
            i_next = (i + 1) % num_circ
            v1_out = outer_offset + j * num_circ + i
            v2_out = outer_offset + j * num_circ + i_next
            v3_out = outer_offset + (j + 1) * num_circ + i_next
            v4_out = outer_offset + (j + 1) * num_circ + i
            faces.append([v1_out, v4_out, v2_out])
            faces.append([v2_out, v4_out, v3_out])
    
    # Bottom cap
    j_bottom = 0
    for i in range(num_circ):
        i_next = (i + 1) % num_circ
        v1_in = 0 * num_circ + i
        v2_in = 0 * num_circ + i_next
        v1_out = outer_offset + j_bottom * num_circ + i
        v2_out = outer_offset + j_bottom * num_circ + i_next
        faces.append([v1_in, v2_out, v2_in])
        faces.append([v1_in, v1_out, v2_out])
    
    # Top cap
    j_top = num_rings_outer - 1
    for i in range(num_circ):
        i_next = (i + 1) % num_circ
        v1_in = 1 * num_circ + i
        v2_in = 1 * num_circ + i_next
        v1_out = outer_offset + j_top * num_circ + i
        v2_out = outer_offset + j_top * num_circ + i_next
        faces.append([v1_in, v2_in, v2_out])
        faces.append([v1_in, v2_out, v1_out])
    
    # Use uint32 for face indices
    faces = np.array(faces, dtype=np.uint32)
    
    return vertices, faces

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
    diameter=None,  # New parameter: diameter in mm
    radius=None,  # Deprecated: kept for backward compatibility, will be converted from diameter
    brim_diameter=None,  # New parameter: brim diameter in mm
    brim_radius=None,  # Deprecated: kept for backward compatibility
    brim_slope_height=None,
    brim_overhang_angle=None,
    top_brim=True,
    bottom_brim=True,
    num_spokes=0,
    spoke_height=None,
    spoke_thickness=None,
    hub_diameter=None,
    hub_height=None,
    progress_callback=None
):
    """
    Generates a straight tube at specified resolution, then procedurally displaces
    the outer surface to form top and bottom brims with a controllable slope.
    Optionally adds spokes connecting to a central hub tube.
    
    Parameters:
    - num_spokes: number of spokes (0 = no spokes)
    - spoke_height: height of spokes in mm (defaults to lamp height if None)
    - spoke_thickness: thickness/width of spokes in mm (defaults to thickness if None)
    - hub_diameter: diameter of hub tube in mm (defaults to 10.0 mm if None)
    - hub_height: height of hub tube in mm (defaults to spoke_height if None)
    """
    if progress_callback:
        progress_callback(2.0, "Starting base lamp generation (tube + brim displacement)...")

    # Inputs from UI/API:
    # - height, diameter (or radius for backward compatibility), brim_diameter (or brim_radius), brim_height are in mm
    # - thickness is in mm
    # Internally we operate in mm to match mm_per_pixel grid resolution
    height_mm = float(height)
    
    # Handle diameter/radius conversion (prefer diameter, fallback to radius for backward compatibility)
    if diameter is not None:
        radius_mm = float(diameter) / 2.0
    elif radius is not None:
        radius_mm = float(radius)
    else:
        raise ValueError("Either diameter or radius must be provided")
    
    # Handle brim_diameter/brim_radius conversion
    if brim_diameter is not None:
        brim_radius_mm = float(brim_diameter) / 2.0
    elif brim_radius is not None:
        brim_radius_mm = float(brim_radius)
    else:
        raise ValueError("Either brim_diameter or brim_radius must be provided")
    
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

    # Add spokes if requested
    num_spokes_int = int(num_spokes) if num_spokes else 0
    if num_spokes_int > 0:
        if progress_callback:
            progress_callback(90.0, "Adding spokes and hub...")
        
        # Default values for spoke parameters
        spoke_height_mm = float(spoke_height) if spoke_height is not None else float(height_mm)
        spoke_thickness_mm = float(spoke_thickness) if spoke_thickness is not None else thickness_mm
        hub_diameter_mm = float(hub_diameter) if hub_diameter is not None else 10.0
        hub_radius_mm = hub_diameter_mm / 2.0
        # Hub height defaults to spoke height if not specified
        hub_height_mm = float(hub_height) if hub_height is not None else spoke_height_mm
        
        # Position hub at bottom of lamp (z=0)
        hub_bottom_center = np.array([0.0, 0.0, 0.0])
        
        # Create hub tube (starts at bottom and goes up)
        hub_vertices, hub_faces = _create_hub_tube(
            center_pos=hub_bottom_center,
            height=hub_height_mm,
            radius=hub_radius_mm,
            mm_per_pixel=mm_per_pixel,
            thickness=thickness_mm,
            progress_callback=progress_callback
        )
        
        # Create hub mesh
        hub_mesh = trimesh.Trimesh(vertices=hub_vertices, faces=hub_faces, process=False)
        
        # Combine hub with main mesh
        all_meshes = [displaced_mesh, hub_mesh]
        
        # Generate spokes at bottom (z=0 to z=spoke_height_mm)
        # Spokes should extend into the hub to ensure connection
        spoke_angle_step = 2.0 * np.pi / num_spokes_int
        for i in range(num_spokes_int):
            angle = i * spoke_angle_step
            
            # Start position: on outer surface of lamp at bottom (z=0)
            start_x = radius_mm * np.cos(angle)
            start_y = radius_mm * np.sin(angle)
            start_z = 0.0
            start_pos = np.array([start_x, start_y, start_z])
            
            # End position: extend into the hub (past the outer radius, to the inner radius or center)
            # This ensures the spoke connects to the hub
            # Extend to hub inner radius (hub_radius - thickness) or at least to center
            hub_inner_radius = max(0.0, hub_radius_mm - thickness_mm)
            # Extend slightly past inner radius to ensure connection
            end_radius = max(0.0, hub_inner_radius - spoke_thickness_mm * 0.5)
            end_x = end_radius * np.cos(angle)
            end_y = end_radius * np.sin(angle)
            end_z = 0.0
            end_pos = np.array([end_x, end_y, end_z])
            
            # Create rectangular spoke
            spoke_vertices, spoke_faces = _create_spoke(
                start_pos=start_pos,
                end_pos=end_pos,
                thickness=spoke_thickness_mm,
                height=spoke_height_mm,
                mm_per_pixel=mm_per_pixel,
                progress_callback=progress_callback
            )
            
            if len(spoke_vertices) > 0:
                spoke_mesh = trimesh.Trimesh(vertices=spoke_vertices, faces=spoke_faces, process=False)
                all_meshes.append(spoke_mesh)
            
            # Update progress periodically for spokes
            if progress_callback:
                spoke_progress = 90.0 + ((i + 1) / num_spokes_int) * 5.0  # 90-95% range
                progress_callback(spoke_progress, f"Generating spoke {i+1}/{num_spokes_int}...")
        
        # Combine all meshes
        combined_mesh = trimesh.util.concatenate(all_meshes)
        combined_mesh.fix_normals()
        
        # Clean up intermediate meshes
        del all_meshes, displaced_mesh, hub_mesh
        if 'spoke_mesh' in locals():
            del spoke_mesh
        gc.collect()
        
        if progress_callback:
            progress_callback(95.0, "Finalizing mesh data...")
        
        # Ensure output arrays are memory-efficient types
        return {
            'vertices': combined_mesh.vertices.astype(np.float32).copy(),
            'faces': combined_mesh.faces.astype(np.uint32).copy()
        }

    if progress_callback:
        progress_callback(95.0, "Finalizing mesh data...")

    # Ensure output arrays are memory-efficient types
    return {
        'vertices': displaced_mesh.vertices.astype(np.float32).copy(),
        'faces': displaced_mesh.faces.astype(np.uint32).copy()
    }