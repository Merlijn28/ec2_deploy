"""
Mesh Export
Exports meshes to STL and GLB formats
"""

import trimesh
from pathlib import Path


def export_stl(mesh: trimesh.Trimesh, filepath: str) -> None:
    """
    Export mesh to STL format
    
    Args:
        mesh: Mesh to export
        filepath: Output file path
    """
    try:
        # Ensure mesh is valid
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        
        # Try to fill holes if mesh is not watertight (requires networkx)
        # This is optional - STL export works fine without it
        try:
            if not mesh.is_volume:
                mesh.fill_holes()
        except (ImportError, AttributeError, Exception):
            # If fill_holes fails (e.g., networkx not available), continue anyway
            # STL export doesn't require watertight meshes
            pass
        
        # Export as STL
        mesh.export(filepath, file_type='stl')
    except Exception as e:
        raise ValueError(f"Failed to export STL: {e}")


def export_glb(mesh: trimesh.Trimesh, filepath: str) -> None:
    """
    Export mesh to GLB format
    
    Args:
        mesh: Mesh to export
        filepath: Output file path
    """
    try:
        # Ensure mesh is valid
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        
        # Fix normals if needed
        if mesh.vertex_normals is None:
            mesh.fix_normals()
        
        # Export as GLB
        # GLB is a binary GLTF format
        mesh.export(filepath, file_type='glb')
    except Exception as e:
        raise ValueError(f"Failed to export GLB: {e}")
