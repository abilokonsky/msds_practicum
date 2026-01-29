""" T92 Data Generation - FAST VERSION
    EEVEE rendering, parallel safe
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
# MEMORY MANAGEMENT
# ============================================
def cleanup_memory():
    for img in list(bpy.data.images):
        if img.users == 0 or 'Render' in img.name:
            try:
                bpy.data.images.remove(img)
            except:
                pass
    
    for mat in list(bpy.data.materials):
        if mat.users == 0 or mat.name.startswith('TankColor_'):
            try:
                bpy.data.materials.remove(mat)
            except:
                pass
    
    for world in list(bpy.data.worlds):
        if world.users == 0:
            try:
                bpy.data.worlds.remove(world)
            except:
                pass
    
    gc.collect()


# ============================================
# AUTO-DETECT ELEVATION
# ============================================
def get_elevation_from_blend():
    blend_path = Path(bpy.data.filepath)
    match = re.search(r't92_(\d{2})', blend_path.stem.lower())
    if match:
        elevation = int(match.group(1))
        print(f"Elevation: {elevation}° from {blend_path.name}")
        return elevation
    return None


# ============================================
# RESUME SUPPORT
# ============================================
def get_resume_index(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    
    existing = list(output_dir.glob('img_*.png'))
    if not existing:
        return 0
    
    max_idx = -1
    for img in existing:
        match = re.search(r'img_(\d+)_', img.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    
    print(f"Resuming from index {max_idx + 1}")
    return max_idx + 1


# ============================================
# CAMERA
# ============================================
def spherical_to_cartesian(distance, azimuth_deg, zenith_deg):
    az_rad = math.radians(azimuth_deg)
    zen_rad = math.radians(zenith_deg)
    
    h = distance * math.cos(zen_rad)
    v = distance * math.sin(zen_rad)
    
    return Vector((h * math.cos(az_rad), -h * math.sin(az_rad), v))


def setup_camera(camera, target, azimuth, zenith, distance):
    camera.location = target + spherical_to_cartesian(distance, azimuth, zenith)
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
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
    
    env = tree.nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(str(hdri_path))
    
    bg = tree.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = random.uniform(0.7, 1.3)
    
    out = tree.nodes.new('ShaderNodeOutputWorld')
    
    tree.links.new(env.outputs['Color'], bg.inputs['Color'])
    tree.links.new(bg.outputs['Background'], out.inputs['Surface'])


# ============================================
# COLOR RANDOMIZATION
# ============================================
def randomize_tank_color(tank):
    if not tank.data:
        return
    
    r = random.uniform(0.15, 0.65)
    g = random.uniform(0.20, 0.60)
    b = random.uniform(0.15, 0.55)
    
    mat = bpy.data.materials.new(name=f"TankColor_{random.randint(0, 99999)}")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
    bsdf.inputs['Roughness'].default_value = random.uniform(0.4, 0.8)
    bsdf.inputs['Metallic'].default_value = random.uniform(0.0, 0.2)
    
    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    if len(tank.data.materials) == 0:
        tank.data.materials.append(mat)
    else:
        for i in range(len(tank.data.materials)):
            tank.data.materials[i] = mat


# ============================================
# MAIN
# ============================================
def run(num_images=NUM_IMAGES):
    elevation_angle = get_elevation_from_blend()
    if elevation_angle is None:
        print("ERROR: Could not detect elevation")
        return
    
    output_dir = OUTPUT_BASE / f'elev_{elevation_angle:02d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_idx = get_resume_index(output_dir)
    if start_idx >= num_images:
        print("Already complete!")
        return
    
    # Find objects
    camera = bpy.data.objects.get('Camera')
    tank = None
    for name in [f'T92_{elevation_angle:02d}', 'T92', 'Tank']:
        if name in bpy.data.objects:
            tank = bpy.data.objects[name]
            break
    
    if not camera or not tank:
        print("ERROR: Missing camera or tank")
        return
    
    target = tank.location + Vector((0, 0, TARGET_HEIGHT_OFFSET))
    original_rot = tank.rotation_euler.copy()
    
    # ========== EEVEE SETTINGS (FAST) ==========
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 32
    bpy.context.scene.render.resolution_x = RESOLUTION[0]
    bpy.context.scene.render.resolution_y = RESOLUTION[1]
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = False
    
    # Assets
    asset_dir = Path(bpy.data.filepath).parent
    hdri_paths = list(asset_dir.glob('hdr/*.hdr')) + list(asset_dir.glob('hdr/*.exr'))
    
    remaining = num_images - start_idx
    print(f"\nGenerating {remaining} images for elevation {elevation_angle}°\n")
    
    # CSV
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
        az = random.uniform(0, 360)
        zen = random.uniform(ZENITH_MIN, ZENITH_MAX)
        dist = random.uniform(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX)
        
        tank.rotation_euler = original_rot.copy()
        tank.rotation_euler.z = original_rot.z - math.radians(az)
        
        cam_pos = setup_camera(camera, target, 0, zen, dist)
        
        if hdri_paths:
            load_hdri(random.choice(hdri_paths))
        
        randomize_tank_color(tank)
        
        filename = f'img_{idx:06d}_a{az:06.2f}_e{elevation_angle:02d}_z{zen:05.2f}.png'
        bpy.context.scene.render.filepath = str(output_dir / filename)
        bpy.ops.render.render(write_still=True)
        
        writer.writerow([
            filename, idx, round(az, 2), elevation_angle, round(zen, 2),
            round(dist, 2), round(cam_pos.x, 2), round(cam_pos.y, 2), round(cam_pos.z, 2)
        ])
        
        done = idx - start_idx + 1
        if done % 50 == 0:
            print(f"Progress: {done}/{remaining} ({done/remaining*100:.0f}%)")
            cleanup_memory()
            csv_file.flush()
    
    csv_file.close()
    tank.rotation_euler = original_rot
    print(f"\n✓ Complete: {num_images} images\n")


run()