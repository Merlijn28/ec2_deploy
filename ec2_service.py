"""
EC2 mesh generation service entry point
Generates lamp meshes with progress updates to S3
"""

import json
import os
import sys
from pathlib import Path

# Add python/ directory to Python path for dependencies
# This allows imports like 'import trimesh' to find packages in python/
# Service handler is at the root of the deployment package

# Use the configured task root for EC2 if available, otherwise use the module directory
_ec2_task_root = os.environ.get('EC2_TASK_ROOT')
if _ec2_task_root:
    _service_dir = Path(_ec2_task_root)
else:
    _service_dir = Path(__file__).parent.resolve()

_python_dir = _service_dir / 'python'

# Add python directory to path if it exists and isn't already there
if _python_dir.exists() and _python_dir.is_dir():
    python_path = str(_python_dir.resolve())
    # Insert at the beginning to prioritize our packages
    if python_path not in sys.path:
        sys.path.insert(0, python_path)
    # Also ensure the parent directory is in path for absolute imports
    service_path = str(_service_dir.resolve())
    if service_path not in sys.path:
        sys.path.insert(0, service_path)

import boto3
import tempfile
import trimesh
import numpy as np
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')
s3_bucket = os.getenv('S3_BUCKET_NAME', 'heroku-lumo-storage')

# Import mesh generation modules
from meshgen.base_lamp import create_base_lamp
from meshgen.displace import apply_displacement
from meshgen.export import export_glb, export_stl


def update_progress(job_id: str, status: str, progress: float, message: str, 
                   stage: Optional[str] = None, vertices: Optional[int] = None,
                   faces: Optional[int] = None, error: Optional[str] = None,
                   parameters: Optional[Dict[str, Any]] = None,
                   average_brightness: Optional[float] = None,
                   processed_image_b64: Optional[str] = None,
                   original_image_b64: Optional[str] = None,
                   image_width: Optional[int] = None,
                   image_height: Optional[int] = None,
                   target_width: Optional[int] = None,
                   target_height: Optional[int] = None):
    """
    Update progress file in S3
    
    Args:
        job_id: Job identifier
        status: Job status ('pending', 'running', 'completed', 'failed')
        progress: Progress percentage (0.0-100.0)
        message: Status message
        stage: Current processing stage
        vertices: Number of vertices (if available)
        faces: Number of faces (if available)
        error: Error message (if failed)
        parameters: Generation parameters to store (for light test mode)
        average_brightness: Average brightness of the processed image (0.0-1.0)
        processed_image_b64: Base64 encoded processed image (PNG)
        original_image_b64: Base64 encoded original image (PNG)
        image_width: Width of the source image in pixels
        image_height: Height of the source image in pixels
        target_width: Target resolution width (if known)
        target_height: Target resolution height (if known)
    """
    progress_key = f"jobs/{job_id}/progress.json"
    
    # Try to read existing progress to preserve parameters
    existing_params = None
    try:
        existing_response = s3_client.get_object(Bucket=s3_bucket, Key=progress_key)
        existing_data = json.loads(existing_response['Body'].read().decode('utf-8'))
        existing_params = existing_data.get('parameters')
    except:
        pass  # File doesn't exist yet, that's fine
    
    progress_data = {
        'status': status,
        'progress': float(progress),
        'message': message,
        'stage': stage,
        'vertices': vertices,
        'faces': faces
    }
    if error:
        progress_data['error'] = error
    
    # Store parameters (use new ones if provided, otherwise keep existing)
    if parameters is not None:
        progress_data['parameters'] = parameters
    elif existing_params is not None:
        progress_data['parameters'] = existing_params

    if average_brightness is not None:
        progress_data['average_brightness'] = average_brightness
    if processed_image_b64 is not None:
        progress_data['processed_image_b64'] = processed_image_b64
    if original_image_b64 is not None:
        progress_data['original_image_b64'] = original_image_b64
    if image_width is not None:
        progress_data['image_width'] = image_width
    if image_height is not None:
        progress_data['image_height'] = image_height
    if target_width is not None:
        progress_data['target_width'] = target_width
    if target_height is not None:
        progress_data['target_height'] = target_height
    
    try:
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=progress_key,
            Body=json.dumps(progress_data),
            ContentType='application/json'
        )
        logger.info(f"Progress updated: {status} - {progress}% - {message}")
    except Exception as e:
        logger.error(f"Failed to update progress: {e}")


def download_from_s3(s3_key: str) -> bytes:
    """Download file from S3"""
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        raise ValueError(f"Failed to download from S3: {e}")


def upload_to_s3(file_path: str, s3_key: str):
    """Upload file to S3"""
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Preparing to upload {file_size} bytes to S3: s3://{s3_bucket}/{s3_key}")
        
        # Determine content type
        ext = s3_key.lower().split('.')[-1]
        content_types = {
            'glb': 'model/gltf-binary',
            'stl': 'model/stl',
            'json': 'application/json',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg'
        }
        content_type = content_types.get(ext, 'application/octet-stream')
        
        logger.info(f"Uploading to S3 bucket: {s3_bucket}, key: {s3_key}, content-type: {content_type}")
        
        s3_client.upload_file(
            file_path,
            s3_bucket,
            s3_key,
            ExtraArgs={'ContentType': content_type}
        )
        
        # Verify upload by checking if file exists in S3
        try:
            s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
            logger.info(f"Successfully uploaded and verified: s3://{s3_bucket}/{s3_key}")
        except Exception as verify_error:
            logger.warning(f"Upload completed but verification failed: {verify_error}")
            # Don't fail the upload if verification fails - upload might have succeeded
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}", exc_info=True)
        raise ValueError(f"Failed to upload to S3: {e}")


def ec2_handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    EC2 service entry point for mesh generation
    
    Expected event structure:
    {
        'input_params': {
            'lamp_params': {...},  # For base mesh generation
            'base_mesh_s3_key': '...',  # OR: S3 key to base mesh
            'base_mesh_data': {...},  # OR: Serialized mesh data
            'image_s3_key': '...',  # Optional: S3 key to displacement image
            'image_base64': '...',  # OR: Base64 encoded image
            'intensity': 0.5,
            'scale': None,
            'offset': 0.0,
            'format': 'glb',
            'target_width': None,
            'target_height': None,
            'thickness_mm': 3.0,
            'min_clearance_mm': 0.6,
            'smoothing': 1.0
        },
        'output_s3_key': 'jobs/{job_id}/mesh.glb',
        'job_id': '...'
    }
    """
    try:
        # Log S3 configuration
        logger.info(f"S3 Configuration - Bucket: {s3_bucket}, Region: {s3_client.meta.region_name}")
        
        # Extract parameters
        input_params = event.get('input_params', {})
        output_s3_key = event.get('output_s3_key')
        job_id = event.get('job_id')
        
        if not job_id:
            raise ValueError("job_id is required")
        if not output_s3_key:
            raise ValueError("output_s3_key is required")
        
        # Ensure output_s3_key is in the correct format (jobs/{job_id}/mesh.{format})
        # If it's in the old format (jobs/{job_id}.{format}), fix it
        if output_s3_key.startswith(f"jobs/{job_id}.") and "/" not in output_s3_key.replace(f"jobs/{job_id}.", ""):
            # Old format detected: jobs/{job_id}.{format} - convert to new format
            format_ext = output_s3_key.split('.')[-1]
            output_s3_key = f"jobs/{job_id}/mesh.{format_ext}"
            logger.warning(f"Converted old format output path to new format: {output_s3_key}")
        
        logger.info(f"Starting mesh generation for job {job_id}")
        logger.info(f"Output will be saved to: s3://{s3_bucket}/{output_s3_key}")
        
        # Note: We don't check bucket accessibility with head_bucket here because:
        # 1. It requires s3:ListBucket permission which may not be granted
        # 2. The actual S3 operations (get_object, put_object, upload_file) will fail
        #    with proper error messages if there are real permission issues
        # 3. This avoids unnecessary 403 errors when the service has object-level permissions
        
        update_progress(job_id, 'running', 0.0, 'Initializing mesh generation...', 'Initialization')
        
        # Step 1: Load or generate base mesh
        base_mesh = None
        lamp_params = input_params.get('lamp_params')
        base_mesh_s3_key = input_params.get('base_mesh_s3_key')
        base_mesh_data = input_params.get('base_mesh_data')
        
        if lamp_params:
            # Generate base mesh from parameters
            logger.info("Generating base mesh from parameters")
            update_progress(job_id, 'running', 5.0, 'Generating base lamp mesh...', 'Base Mesh Generation')
            
            # Progress callback for mesh generation (maps to 0-60%)
            def mesh_progress_callback(progress_pct: float, message: str):
                overall_progress = progress_pct * 0.6  # Map to 0-60%
                update_progress(job_id, 'running', overall_progress, message, 'Base Mesh Generation')
            
            diameter_param = lamp_params.get('diameter')
            if diameter_param is None:
                raise ValueError("lamp_params must include 'diameter'")
            brim_diameter_param = lamp_params.get('brim_diameter', 4.0)
            
            mesh_data = create_base_lamp(
                height=lamp_params.get('height', 200.0),
                diameter=diameter_param,
                mm_per_pixel=lamp_params.get('mm_per_pixel', 0.25),
                thickness=lamp_params.get('thickness', 3.0),
                brim_diameter=brim_diameter_param,
                brim_height=lamp_params.get('brim_height', 4.0),
                brim_overhang_angle=lamp_params.get('brim_overhang_angle', 45.0),
                top_brim=lamp_params.get('top_brim', True),
                bottom_brim=lamp_params.get('bottom_brim', True),
                progress_callback=mesh_progress_callback
            )
            
            base_mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'],
                faces=mesh_data['faces'],
                process=True
            )
            
            update_progress(
                job_id, 
                'running', 
                60.0, 
                f'Base mesh generated ({len(base_mesh.vertices):,} vertices, {len(base_mesh.faces):,} faces)',
                'Base Mesh Complete',
                vertices=len(base_mesh.vertices),
                faces=len(base_mesh.faces)
            )
            
        elif base_mesh_s3_key:
            # Load base mesh from S3
            logger.info(f"Loading base mesh from S3: {base_mesh_s3_key}")
            update_progress(job_id, 'running', 10.0, 'Downloading base mesh from S3...', 'Loading Base Mesh')
            
            mesh_data = download_from_s3(base_mesh_s3_key)
            
            # Save to temp file and load with trimesh
            with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as temp_file:
                temp_file.write(mesh_data)
                temp_file_path = temp_file.name
            
            try:
                loaded = trimesh.load(temp_file_path)
                if isinstance(loaded, trimesh.Scene):
                    meshes = [geom for geom in loaded.geometry.values() 
                             if isinstance(geom, trimesh.Trimesh)]
                    if len(meshes) == 1:
                        base_mesh = meshes[0]
                    else:
                        base_mesh = trimesh.util.concatenate(meshes)
                else:
                    base_mesh = loaded
                
                update_progress(
                    job_id, 
                    'running', 
                    60.0, 
                    f'Base mesh loaded ({len(base_mesh.vertices):,} vertices)',
                    'Base Mesh Loaded',
                    vertices=len(base_mesh.vertices),
                    faces=len(base_mesh.faces)
                )
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        elif base_mesh_data:
            # Reconstruct mesh from serialized data
            logger.info("Reconstructing mesh from serialized data")
            update_progress(job_id, 'running', 10.0, 'Reconstructing mesh from data...', 'Loading Base Mesh')
            
            vertices = np.array(base_mesh_data['vertices'], dtype=np.float32)
            faces = np.array(base_mesh_data['faces'], dtype=np.uint32)
            base_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            
            update_progress(
                job_id, 
                'running', 
                60.0, 
                f'Mesh reconstructed ({len(base_mesh.vertices):,} vertices)',
                'Base Mesh Loaded',
                vertices=len(base_mesh.vertices),
                faces=len(base_mesh.faces)
            )
        else:
            raise ValueError("Either lamp_params, base_mesh_s3_key, or base_mesh_data must be provided")
        
        # Step 2: Apply displacement if image is provided
        image_s3_key = input_params.get('image_s3_key')
        image_base64 = input_params.get('image_base64')
        intensity = input_params.get('intensity', 0.5)
        avg_brightness = None
        processed_img_b64 = None
        original_img_b64 = None
        img_width = None
        img_height = None
        
        if image_s3_key or image_base64:
            # Apply displacement mapping
            logger.info("Applying displacement mapping")
            update_progress(job_id, 'running', 60.0, 'Preparing displacement mapping...', 'Displacement')
            
            # Load image
            image_path = None
            if image_s3_key:
                update_progress(job_id, 'running', 62.0, 'Downloading displacement image from S3...', 'Displacement')
                image_data = download_from_s3(image_s3_key)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(image_data)
                    image_path = temp_file.name
            elif image_base64:
                import base64
                update_progress(job_id, 'running', 62.0, 'Decoding displacement image...', 'Displacement')
                image_data = base64.b64decode(image_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(image_data)
                    image_path = temp_file.name
            
            try:
                # Displacement progress callback (maps to 60-90%)
                def displacement_progress(p: float, m: str):
                    overall = 60.0 + (p * 0.3)  # 60-90%
                    update_progress(job_id, 'running', overall, m, 'Displacement')
                
                displaced_mesh, avg_brightness, processed_img_b64, original_img_b64, img_width, img_height = apply_displacement(
                    mesh=base_mesh,
                    image_path=image_path,
                    intensity=intensity,
                    scale=input_params.get('scale'),
                    offset=input_params.get('offset', 0.0),
                    target_width=input_params.get('target_width'),
                    target_height=input_params.get('target_height'),
                    thickness_mm=input_params.get('thickness_mm', 3.0),
                    min_clearance_mm=input_params.get('min_clearance_mm', 0.6),
                    smoothing=input_params.get('smoothing', 1.0),
                    progress_callback=displacement_progress
                )
                
                base_mesh = displaced_mesh
                
                update_progress(
                    job_id, 
                    'running', 
                    90.0, 
                    f'Displacement applied ({len(base_mesh.vertices):,} vertices, {len(base_mesh.faces):,} faces)',
                    'Displacement Complete',
                    vertices=len(base_mesh.vertices),
                    faces=len(base_mesh.faces)
                )
            finally:
                if image_path and os.path.exists(image_path):
                    os.unlink(image_path)
        else:
            # No displacement - skip to export
            update_progress(job_id, 'running', 90.0, 'Skipping displacement (no image provided)', 'Ready for Export')
        
        # Step 3: Export mesh
        logger.info("Exporting mesh")
        logger.info(f"S3 bucket: {s3_bucket}, Output S3 key: {output_s3_key}")
        update_progress(job_id, 'running', 90.0, 'Exporting mesh...', 'Export')
        
        output_format = input_params.get('format', 'glb').lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}') as temp_file:
            temp_file_path = temp_file.name
        
        try:
            logger.info(f"Exporting mesh to temporary file: {temp_file_path}")
            if output_format == 'stl':
                export_stl(base_mesh, temp_file_path)
            else:
                export_glb(base_mesh, temp_file_path)
            
            # Verify export file was created
            if not os.path.exists(temp_file_path):
                raise ValueError(f"Export file was not created: {temp_file_path}")
            
            exported_size = os.path.getsize(temp_file_path)
            logger.info(f"Mesh exported successfully: {exported_size} bytes")
            
            update_progress(job_id, 'running', 95.0, 'Uploading result to S3...', 'Upload')
            
            # Upload to S3
            logger.info(f"Starting S3 upload: {output_s3_key}")
            upload_to_s3(temp_file_path, output_s3_key)
            logger.info(f"S3 upload completed: {output_s3_key}")
            
            # Copy image to permanent location if it exists
            permanent_image_s3_key = None
            original_image_s3_key = input_params.get('image_s3_key')
            if original_image_s3_key:
                try:
                    # Determine the file extension from the original key
                    # Handle both .png and other extensions
                    # Extract extension from the end of the key (handles test images like job_id_test.png)
                    key_lower = original_image_s3_key.lower()
                    if key_lower.endswith('.png'):
                        image_ext = 'png'
                    elif key_lower.endswith('.jpg'):
                        image_ext = 'jpg'
                    elif key_lower.endswith('.jpeg'):
                        image_ext = 'jpg'
                    else:
                        image_ext = 'png'  # Default to png
                    
                    # Create permanent image key in jobs/{job_id}/ folder
                    permanent_image_s3_key = f"jobs/{job_id}/image.{image_ext}"
                    
                    # Copy image to permanent location in job folder
                    logger.info(f"Copying image from {original_image_s3_key} to {permanent_image_s3_key}")
                    image_data = download_from_s3(original_image_s3_key)
                    
                    # Upload to permanent location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image_ext}') as temp_image_file:
                        temp_image_file.write(image_data)
                        temp_image_path = temp_image_file.name
                    
                    try:
                        upload_to_s3(temp_image_path, permanent_image_s3_key)
                        logger.info(f"Image copied to permanent location: {permanent_image_s3_key}")
                    finally:
                        if os.path.exists(temp_image_path):
                            os.unlink(temp_image_path)
                except Exception as e:
                    logger.warning(f"Failed to copy image to permanent location: {e}")
                    # Fall back to original location if copy fails
                    permanent_image_s3_key = original_image_s3_key
            
            # Prepare parameters for storage (for light test mode and save lamp functionality)
            stored_parameters = None
            if lamp_params:
                # Store lamp parameters plus displacement parameters if applicable
                stored_parameters = {
                    'height': lamp_params.get('height'),
                    'diameter': diameter_param,
                    'mm_per_pixel': lamp_params.get('mm_per_pixel'),
                    'thickness': lamp_params.get('thickness'),
                    'brim_diameter': lamp_params.get('brim_diameter', brim_diameter_param),
                    'brim_height': lamp_params.get('brim_height'),
                    'brim_overhang_angle': lamp_params.get('brim_overhang_angle'),
                    'top_brim': lamp_params.get('top_brim'),
                    'bottom_brim': lamp_params.get('bottom_brim'),
                    'min_clearance_mm': input_params.get('min_clearance_mm', 0.6),
                    'intensity': input_params.get('intensity'),
                    'smoothing': input_params.get('smoothing', 1.0),
                    'image_s3_key': permanent_image_s3_key or original_image_s3_key  # Use permanent location if available
                }
            elif base_mesh_s3_key or base_mesh_data:
                # If using saved lamp, we don't have the original parameters
                # But we can still store displacement parameters
                stored_parameters = {
                    'thickness': input_params.get('thickness_mm', 3.0),
                    'min_clearance_mm': input_params.get('min_clearance_mm', 0.6),
                    'intensity': input_params.get('intensity'),
                    'smoothing': input_params.get('smoothing', 1.0),
                    'image_s3_key': permanent_image_s3_key or original_image_s3_key  # Use permanent location if available
                }
            
            # Final progress update
            update_progress(
                job_id, 
                'completed', 
                100.0, 
                'Mesh generation completed!',
                'Complete',
                vertices=len(base_mesh.vertices),
                faces=len(base_mesh.faces),
                parameters=stored_parameters,
                average_brightness=avg_brightness,
                processed_image_b64=processed_img_b64,
                original_image_b64=original_img_b64,
                image_width=img_width,
                image_height=img_height,
                target_width=input_params.get('target_width'),
                target_height=input_params.get('target_height')
            )
            
            logger.info(f"Mesh generation completed for job {job_id}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'completed',
                    'job_id': job_id,
                    's3_key': output_s3_key,
                    'vertices': len(base_mesh.vertices),
                    'faces': len(base_mesh.faces)
                })
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in EC2 handler: {error_message}", exc_info=True)
        
        # Update progress with error
        job_id = event.get('job_id', 'unknown')
        update_progress(
            job_id,
            'failed',
            0.0,
            f'Mesh generation failed: {error_message}',
            'Error',
            error=error_message
        )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'failed',
                'job_id': job_id,
                'error': error_message
            })
        }

