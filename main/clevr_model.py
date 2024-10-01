import pyro
from pyro import poutine
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

import subprocess
import os
import json
import math
from PIL import Image
import matplotlib.pyplot as plt

from utils.distributions import MyNormal, MyPoisson
from .setup import params

import warnings
warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger("train")
device = params["device"]

img_transform = transforms.Compose([transforms.ToTensor()])

dir_path = os.path.dirname(__file__)
properties_json_path = os.path.join(dir_path, "clevr_data", "properties.json")
min_objects = 3
max_objects = 10
max_retries = 50
min_dist = 0.25
min_margin = 0.4

def to_int(value: Tensor):
    return int(torch.round(value))

def preprocess_clevr(image, resolution=(128, 128)):
    #image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].
    image = F.interpolate(input=image, size=resolution, mode='bilinear', antialias=True)
    image = torch.clamp(image, 0., 1.)
    return image

def sample_clevr_scene(N):
    
    logger.info("generating clevr scene...")
    
    # Load the property file
    with open(properties_json_path, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
            material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())
    
    
    # Sample scene 
    num_objects = pyro.sample("N", MyPoisson(torch.tensor(5.), validate_args = False), obs=N)
    if type(num_objects) == Tensor: num_objects = to_int(num_objects)

    positions = []
    objects = []
    scene_struct = {'directions': {}}
    
    for i in range(num_objects):

        # Choose a random size
        size = pyro.sample(f"size_{i}", dist.Categorical(probs=torch.tensor([1/len(size_mapping) for _ in range(len(size_mapping))])))
        size_name, r = size_mapping[to_int(size)]

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > max_retries:
                return None
            
            # Choose a random location
            with poutine.block():
              x = pyro.sample(f"x_{i}", dist.Uniform(-3, 3))
              y = pyro.sample(f"y_{i}", dist.Uniform(-3, 3))            
            
            # Assuming the default camera position
            cam_default_pos = [7.358891487121582, -6.925790786743164, 4.958309173583984]
            plane_normal = torch.tensor([0., 0., 1.])
            cam_behind = torch.tensor([-0.6515582203865051, 0.6141704320907593, -0.44527149200439453])
            cam_left = torch.tensor([-0.6859207153320312, -0.7276763916015625, 0.0])
            cam_up = torch.tensor([-0.32401347160339355, 0.3054208755493164, 0.8953956365585327])
            plane_behind = torch.tensor([-0.727676272392273, 0.6859206557273865, 0.0])
            plane_left = torch.tensor([-0.6859206557273865, -0.7276763319969177, 0.0])
            plane_up = torch.tensor([0., 0., 1.])

            # Save all six axis-aligned directions in the scene struct
            scene_struct['directions']['behind'] = tuple(plane_behind)
            scene_struct['directions']['front'] = tuple(-plane_behind)
            scene_struct['directions']['left'] = tuple(plane_left)
            scene_struct['directions']['right'] = tuple(-plane_left)
            scene_struct['directions']['above'] = tuple(plane_up)
            scene_struct['directions']['below'] = tuple(-plane_up)
             
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                distance = math.sqrt(dx * dx + dy * dy)
                if distance - r - rr < min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < min_margin:
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break          
        
        # Choose random color and shape
        shape = pyro.sample(f"shape_{i}", dist.Categorical(probs=torch.tensor([1/len(object_mapping) for _ in range(len(object_mapping))])))
        obj_name, obj_name_out = object_mapping[to_int(shape)]

        color_mapping = list(color_name_to_rgba.items())
        color = pyro.sample(f"color_{i}", dist.Categorical(probs=torch.tensor([1/len(color_mapping) for _ in range(len(color_mapping))])))
        color_name, rgba = color_mapping[to_int(color)]

         # For cube, adjust the size a bit
        if obj_name == 'Cube':
            r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = pyro.sample(f"pose_{i}", dist.Uniform(0., 1.)) * 360. 
        positions.append((x, y, r))

        # Attach a random material
        mat = pyro.sample(f"mat_{i}", dist.Categorical(probs=torch.tensor([1/len(material_mapping) for _ in range(len(material_mapping))])))
        mat_name, mat_name_out = material_mapping[to_int(mat)]
        
        # Append the object attributes to the scene list
        objects.append({
            "shape": obj_name,
            "color": color_name,
            "rgba": rgba,
            "size": r,
            "material": mat_name,
            "pose": theta.item(),
            "position": (x.item(), y.item())
        })
    
    return objects

def generate_blender_script(objects, output_file=os.path.join(dir_path, "clevr_data", "clevr_scene.blend")):
    """
    Generate a Blender Python script to render the CLEVR-like scene.
    """
    script = """
import bpy
import random
import os

from mathutils import Vector

# Set directory path
dir_path = os.path.dirname(__file__)

# Open main file
bpy.ops.wm.open_mainfile(filepath=os.path.join(dir_path, "clevr_data", "base_scene.blend"))

# Load materials
def load_materials(material_dir):
    # Load materials from a directory. We assume that the directory contains .blend
    # files with one material each. The file X.blend has a single NodeTree item named
    # X; this NodeTree item must have a "Color" input that accepts an RGBA value.

    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'): continue
        name = os.path.splitext(fn)[0]
        #print(fn)
        blend_file_path = os.path.join(material_dir, fn)
        #print(blend_file_path)
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            available_materials = data_from.materials
            if available_materials: 
                name = available_materials[0]
                data_to.materials.append(name)
        appended_material = bpy.data.materials.get(name)

load_materials(os.path.join(dir_path, "clevr_data", "materials"))

# Set render arguments so we can get pixel coordinates later.
# We use functionality specific to the CYCLES renderer so BLENDER_RENDER
# cannot be used.
render_args = bpy.context.scene.render
render_args.engine = "CYCLES"
render_args.resolution_x = 320
render_args.resolution_y = 240
render_args.resolution_percentage = 100
cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'CUDA'


# Some CYCLES-specific stuff
bpy.data.worlds['World'].cycles.sample_as_light = True
bpy.context.scene.cycles.blur_glossy = 2.0
bpy.context.scene.cycles.samples = 512
bpy.context.scene.cycles.transparent_min_bounces = 8
bpy.context.scene.cycles.transparent_max_bounces = 8
bpy.context.scene.cycles.device = 'GPU'

# Put a plane on the ground so we can compute cardinal directions
bpy.ops.mesh.primitive_plane_add(size=5)
plane = bpy.context.object

def rand(L):
    return 2.0 * L * (random.random() - 0.5)

for i in range(3):
    bpy.data.objects['Camera'].location[i] += rand(0.5)

# Figure out the left, up, and behind directions along the plane and record
# them in the scene structure
camera = bpy.data.objects['Camera']
plane_normal = plane.data.vertices[0].normal
cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
plane_up = cam_up.project(plane_normal).normalized()

# Delete the plane; we only used it for normals anyway. The base scene file
# contains the actual ground plane.
# utils.delete_object(plane)

# Add random jitter to lamp positions
for i in range(3):
    bpy.data.objects['Lamp_Key'].location[i] += rand(1.0)
    bpy.data.objects['Lamp_Back'].location[i] += rand(1.0)
    bpy.data.objects['Lamp_Fill'].location[i] += rand(1.0)

def add_material(name, **properties):
  
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    obj = bpy.context.active_object
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )
    
def add_object(object_dir, name, scale, loc, theta=0):
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))    

# Add objects to the scene
def _add_object(object_dir):
    
    shape_dir = os.path.join(dir_path, "clevr_data", "shapes")
    add_object(shape_dir, object_dir["shape"], object_dir["size"], object_dir["position"], object_dir["pose"])
    
    # Get reference to the object
    obj = bpy.context.object
    
    # Add material for the object
    add_material(object_dir['material'], Color=object_dir['rgba'])

# Sampled objects from Pyro
objects = {}
"""
    
    # Insert the sampled objects
    for i, obj in enumerate(objects):
        script += f"""
objects[{i}] = {obj}
_add_object(objects[{i}])
"""
    
    script += """

# Set render settings
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = os.path.join(dir_path, "rendered_scene.png")

# Render the scene
bpy.ops.render.render(write_still=True)
    """
    
    # Write the Blender script to a file
    script_file = os.path.join(dir_path, "generate_clevr_scene.py")
    with open(script_file, "w") as f:
        f.write(script)
    
    return script_file

def render_scene_in_blender(blender_script):
    """
    Call Blender to execute the script and render the scene.
    """
    blender_path = "/usr/bin/blender"  # Update this with your Blender path
    subprocess.call([blender_path, "--background", "--python", blender_script])

def clevr_model(observations={"image": torch.zeros((1, 3, 128, 128))}, show='all', save_obs=None, N=None):
    
    # Sample a CLEVR-like scene using Pyro
    clevr_scene = None
    while clevr_scene is None: 
        clevr_scene = sample_clevr_scene(N)
    logger.info(f"Sampled scene: {clevr_scene}")
    
    # Generate the Blender script for the sampled scene
    blender_script = generate_blender_script(clevr_scene)
    
    # Call Blender to render the scene
    render_scene_in_blender(blender_script)

    logger.info("Scene rendered and saved...")

    img = img_transform(Image.open(os.path.join(dir_path, "rendered_scene.png"))).unsqueeze(0) # img shape is (1, 4, 240, 320)

    #plt.imshow(img[0].permute(1, 2, 0).numpy())
    #plt.show()

    logger.info(img.shape)

    proc_img = preprocess_clevr(img) # proc_img shape is (1, 4, 128, 128)

    #plt.imshow(proc_img[0].permute(1, 2, 0).numpy())
    #plt.show()

    logger.info(proc_img.shape)

    with pyro.plate(observations["image"].shape[0]):
        #pyro.sample("image", MyBernoulli(img, validate_args=False).to_event(3), obs=observations["image"])
        likelihood_fn = MyNormal(proc_img, torch.tensor(0.1)).get_dist()
        pyro.sample("image", likelihood_fn.to_event(3), obs=observations["image"])