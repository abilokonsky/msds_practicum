""" T92 Data Generation - Production Script
    GPU rendering, resume support, crash protection
"""

import bpy
import math
import csv
import random
import re
import gc
from pathlib import Path
from mathutils import Vector

# ============================================
# CONFIGURATION
# ============================================
OUTPUT_BASE = Path(r'C:\development\msds_practicum\msds_practicum\data\raw')
RESOLUTION = (224, 224)

CAMERA_DISTANCE_MIN = 50.0
CAMERA_DISTANCE_MAX = 150.0

ZENITH_MIN = 1
ZENITH_MAX = 89

NUM_IMAGES = 3600

TARGET_HEIGHT_OFFSET = 1.0


# ============================================
# GPU SETUP
# ============================================
def setup_gpu():
    """Enable GPU rendering with OptiX (best for RTX cards)"""
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        
        # Try OptiX first (fastest for RTX)
        prefs.compute_device_type = 'OPTIX'
        prefs.get_devices()
        
        # Enable all GPUs
        for device in prefs.devices:
            device.use = True
            print(f"  Enabled: {device.name} ({device.type})")
        
        bpy.context.scene.cycles.device = 'GPU'
        print("GPU rendering enabled (OptiX)")
        return True
        
    except Exception as e:
        print(f"OptiX failed: {e}")
        
        try:
            # Fallback to CUDA
            prefs.compute_device_type = 'CUDA'
            prefs.get_devices()
            
            for device in prefs.devices:
                device.use = True
                print(f"  Enabled: {device.name} ({device.type})")
            
            bpy.context.scene.cycles.device = 'GPU'
            print("GPU rendering enabled (CUDA)")
            return True
            
        except Exception as e2:
            print(f"CUDA failed: {e2}")
            print("Falling back to CPU")
            bpy.context.scene.cycles.device = 'CPU'
            return False


# ============================================
# MEMORY MANAGEMENT
# ============================================
def cleanup_memory():
    """Aggressive memory cleanup to prevent crashes"""
    
    # Clear unused images (biggest memory hog)
    for img in list(bpy.data.images):
        if img.users == 0 or 'Render' in img.name or 'Viewer' in img.name:
            try:
                bpy.data.images.remove(img)
            except:
                pass
    
    # Clear unused materials
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            try:
                bpy.data.materials.remove(mat)
            except:
                pass
    
    # Clear unused worlds (from HDRI loading)
    for world in list(bpy.data.worlds):
        if world.users == 0:
            try:
                bpy.data.worlds.remove(world)
            except:
                pass
    
    # Force garbage collection
    gc.collect()


# ============================================
# AUTO-DETECT ELEVATION
# ============================================
def get_elevation_from_blend():
    blend_path = Path(bpy.data.filepath)
    filename = blend_path.stem
    
    match = re.search(r't92_(\d{2})', filename.lower())
    if match:
        elevation = int(match.group(1))
        print(f"Auto-detected elevation: {elevation}° from {blend_path.name}")
        return elevation
    
    print(f"WARNING: Could not detect elevation from: {filename}")
    return None


# ============================================
# RESUME SUPPORT
# ============================================
def get_resume_index(output_dir):
    """Find where to resume from"""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        return 0
    
    existing_images = list(output_dir.glob('img_*.png'))
    
    if not existing_images:
        return 0
    
    max_idx = -1
    for img_path in existing_images:
        match = re.search(r'img_(\d+)_', img_path.name)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)
    
    resume_idx = max_idx + 1
    print(f"Found {len(existing_images)} existing images, resuming from index {resume_idx}")
    
    return resume_idx


# ============================================
# CAMERA POSITIONING
# ============================================
def spherical_to_cartesian(distance, azimuth_deg, zenith_deg):
    az_rad = math.radians(azimuth_deg)
    zen_rad = math.radians(zenith_deg)
    
    horizontal_dist = distance * math.cos(zen_rad)
    vertical_dist = distance * math.sin(zen_rad)
    
    x = horizontal_dist * math.cos(az_rad)
    y = -horizontal_dist * math.sin(az_rad)
    z = vertical_dist
    
    return Vector((x, y, z))


def point_camera_at_target(camera, target_location):
    direction = target_location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def setup_camera(camera, target, azimuth_deg, zenith_deg, distance):
    offset = spherical_to_cartesian(distance, azimuth_deg, zenith_deg)
    camera.location = target + offset
    point_camera_at_target(camera, target)
    return camera.location.copy()


# ============================================
# LIGHTING
# ============================================
def load_hdri(hdri_path):
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    
    env_tex = tree.nodes.new('ShaderNodeTexEnvironment')
    env_tex.image = bpy.data.images.load(str(hdri_path))
    
    background = tree.nodes.new('ShaderNodeBackground')
    background.inputs['Strength'].default_value = random.uniform(0.7, 1.3)
    
    output = tree.nodes.new('ShaderNodeOutputWorld')
    
    tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])
    tree.links.new(background.outputs['Background'], output.inputs['Surface'])
    
    bpy.context.scene.render.film_transparent = False


# ============================================
# COLOR RANDOMIZATION
# ============================================
def randomize_tank_color(tank):
    if not tank.data.materials:
        return
    
    for mat in tank.data.materials:
        if not mat.use_nodes:
            continue
        
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                r = random.uniform(0.2, 0.6)
                g = random.uniform(0.3, 0.6)
                b = random.uniform(0.2, 0.5)
                node.inputs['Base Color'].default_value = (r, g, b, 1.0)
                break


# ============================================
# MAIN GENERATION
# ============================================
def run(elevation_angle=None, num_images=NUM_IMAGES):
    """Generate dataset with GPU rendering and resume support"""
    
    # Auto-detect elevation
    if elevation_angle is None:
        elevation_angle = get_elevation_from_blend()
        if elevation_angle is None:
            print("ERROR: Could not determine elevation.")
            return
    
    # Output directory
    output_dir = OUTPUT_BASE / f'elev_{elevation_angle:02d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume check
    start_idx = get_resume_index(output_dir)
    
    if start_idx >= num_images:
        print(f"Already complete! {start_idx} images exist.")
        return
    
    remaining = num_images - start_idx
    
    # Find objects
    camera = bpy.data.objects.get('Camera')
    if not camera:
        print("ERROR: No camera found")
        return
    
    tank = None
    for name in [f'T92_{elevation_angle:02d}', 'T92', 'Tank']:
        if name in bpy.data.objects:
            tank = bpy.data.objects[name]
            print(f"Found tank: {name}")
            break
    
    if not tank:
        meshes = [o.name for o in bpy.data.objects if o.type == 'MESH']
        print(f"ERROR: Tank not found. Available: {meshes}")
        return
    
    # Setup
    target = tank.location + Vector((0, 0, TARGET_HEIGHT_OFFSET))
    tank_original_rot = tank.rotation_euler.copy()
    
    # Render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.resolution_x = RESOLUTION[0]
    bpy.context.scene.render.resolution_y = RESOLUTION[1]
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = False
    
    # Enable GPU
    print("\nSetting up GPU...")
    setup_gpu()
    
    # Assets
    asset_dir = Path(bpy.data.filepath).parent
    hdri_paths = list(asset_dir.glob('hdr/*.hdr')) + list(asset_dir.glob('hdr/*.exr'))
    
    print(f"\n{'='*60}")
    print(f"T92 DATA GENERATION")
    print(f"{'='*60}")
    print(f"Elevation: {elevation_angle}°")
    print(f"Output: {output_dir}")
    print(f"Resuming from: {start_idx}")
    print(f"Target: {num_images}")
    print(f"Remaining: {remaining}")
    print(f"HDRIs: {len(hdri_paths)}")
    print(f"{'='*60}\n")
    
    # CSV - append or create
    csv_path = output_dir / 'labels.csv'
    
    if start_idx == 0:
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'idx', 'azimuth', 'elevation', 'zenith', 'distance', 'cam_x', 'cam_y', 'cam_z'])
    else:
        csv_file = open(csv_path, 'a', newline='')
        writer = csv.writer(csv_file)
    
    # Main loop
    for idx in range(start_idx, num_images):
        # Random parameters
        azimuth = random.uniform(0, 360)
        zenith = random.uniform(ZENITH_MIN, ZENITH_MAX)
        distance = random.uniform(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX)
        
        # Reset tank
        tank.rotation_euler = tank_original_rot.copy()
        tank.rotation_euler.z = tank_original_rot.z - math.radians(azimuth)
        
        # Position camera
        cam_pos = setup_camera(camera, target, 0, zenith, distance)
        
        # Lighting
        if hdri_paths:
            load_hdri(random.choice(hdri_paths))
        
        # Randomize color
        randomize_tank_color(tank)
        
        # Render
        filename = f'img_{idx:06d}_a{azimuth:06.2f}_e{elevation_angle:02d}_z{zenith:05.2f}.png'
        bpy.context.scene.render.filepath = str(output_dir / filename)
        bpy.ops.render.render(write_still=True)
        
        # Write label
        writer.writerow([
            filename, idx, round(azimuth, 2), elevation_angle, round(zenith, 2),
            round(distance, 2), round(cam_pos.x, 2), round(cam_pos.y, 2), round(cam_pos.z, 2)
        ])
        
        # Progress and cleanup every 25 images
        completed = idx - start_idx + 1
        if completed % 25 == 0:
            pct = completed / remaining * 100
            print(f"Progress: {completed}/{remaining} ({pct:.1f}%) | Total: {idx + 1}/{num_images}")
            cleanup_memory()
            csv_file.flush()
    
    csv_file.close()
    tank.rotation_euler = tank_original_rot
    
    print(f"\n{'='*60}")
    print(f"✓ COMPLETE: {num_images} images")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")


# ============================================
# RUN
# ============================================
run()