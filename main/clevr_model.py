import pyro
from pyro import poutine
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
import numpy as np
from scipy.stats import qmc

import subprocess
import multiprocessing as mp
import os
import json
import math
from PIL import Image
import matplotlib.pyplot as plt
import time
import glob
from torchvision import transforms

from utils.distributions import MyBernoulli, MyNormal
from .setup import params, JOB_SPLIT

import warnings
warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger(params['running_type'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([transforms.ToTensor()])

dir_path = os.path.dirname(__file__)
properties_json_path = os.path.join(dir_path, "clevr_data", "properties.json")
min_objects = 3
max_objects = params['max_objects']
max_retries = 50
min_dist = 0.25
min_margin = 0.4

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
    color_mapping = list(color_name_to_rgba.items())

# imgs_path = os.path.join(dir_path, str(params['jobID']))
# if not os.path.isdir: os.mkdir(imgs_path)

def sample_loc(i):
    with pyro.poutine.block():     
        x_mu = pyro.sample(f"x_{i}", dist.Uniform(-3., 3.))
        y_mu = pyro.sample(f"y_{i}", dist.Uniform(-3., 3.))
    return x_mu, y_mu

def to_int(value: Tensor):
    return int(torch.round(value))

transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
    ]
    )


def preprocess_clevr(image, resolution=params['resolution']):
    
    # logger.info(torch.amin(image))
    # logger.info(torch.amax(image))

    image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].
    image = F.interpolate(input=image, size=resolution, mode='bilinear', antialias=True)
    image = torch.clamp(image, -1., 1.)

    # logger.info(torch.amin(image))
    # logger.info(torch.amax(image))
    
    return image

def get_size_mapping(size):
    return size_mapping[int(size)]
    
def get_shape_mapping(shape):
    return object_mapping[int(shape)]

def get_color_mapping(color):
    return color_mapping[int(color)]

def get_mat_mapping(mat):
    return material_mapping[int(mat)]

def sample_clevr_scene(llh_uncertainty, pixel_coords=None):
    
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
    scene_struct = {'directions': {}}
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)    

    B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']
    M = max_objects 
    
    # Sample the mask to predict real objects
    objects_mask = pyro.sample(f"mask", dist.Bernoulli(0.5).expand([B, M])).to(torch.bool)
    if params['running_type'] == 'eval': 
        if objects_mask.dim() > 2: objects_mask = torch.flatten(objects_mask, 0, 1)

    
    num_objects = torch.sum(objects_mask, dim=-1)
    scenes = []

    # Choose random color and shape
    with pyro.poutine.mask(mask=objects_mask):
        shape = pyro.sample(f"shape", dist.Categorical(probs=torch.tensor([1/len(object_mapping) for _ in range(len(object_mapping))])).expand([B, M]))
        color = pyro.sample(f"color", dist.Categorical(probs=torch.tensor([1/len(color_mapping) for _ in range(len(color_mapping))])).expand([B, M]))
        with pyro.poutine.block():
            theta = pyro.sample(f"pose", dist.Uniform(0., 1.).expand([B, M])) * 360. 
            
            logger.info(theta)

        mat = pyro.sample(f"mat", dist.Categorical(probs=torch.tensor([1/len(material_mapping) for _ in range(len(material_mapping))])).expand([B, M]))
        size = pyro.sample(f"size", dist.Categorical(probs=torch.tensor([1/len(size_mapping) for _ in range(len(size_mapping))])).expand([B, M]))
        
        if params['running_type'] == 'eval':
            if shape.dim() > 2: shape = torch.flatten(shape, 0, 1)
            if color.dim() > 2: color = torch.flatten(color, 0, 1)
            if theta.dim() > 2: theta = torch.flatten(theta, 0, 1)
            if mat.dim() > 2: mat = torch.flatten(mat, 0, 1)

    shape_mapping_list = {}
    color_mapping_list = {}
    mat_mapping_list = {}
    size_mapping_list = {}
    for b in range(B):
        shape_mapping_list[b] = list(map(get_shape_mapping, shape[b].tolist())) # list of tuples [('name', value)]
        color_mapping_list[b] = list(map(get_color_mapping, color[b].tolist())) # list of tuples [('name', value)]
        mat_mapping_list[b] = list(map(get_mat_mapping, mat[b].tolist()))
        size_mapping_list[b] = list(map(get_size_mapping, size[b].tolist()))
    
    obj_name = {}
    color_name = {}
    rgba = {}
    mat_name = {}

    r = {}
    for b in range(B):
        obj_name[b] = [e[0] for e in shape_mapping_list[b]]
        color_name[b] = [e[0] for e in color_mapping_list[b]]
        rgba[b] = [e[1] for e in color_mapping_list[b]]
        mat_name[b] = [e[0] for e in mat_mapping_list[b]]
        r[b] = [e[1] for e in size_mapping_list[b]]

    
    # SAMPLE POSITIONS
    x_min, x_max = -3. + min_margin, 3. - min_margin
    y_min, y_max = -3. + min_margin, 3. - min_margin
    sampling_radius = 0.3
    positions = []
    ncandidates = 100
    for b in range(B):
        engine = qmc.PoissonDisk(d=2, radius=sampling_radius, ncandidates=int(ncandidates))
        pre_positions = engine.random(num_objects[b])

        while pre_positions.shape[0] < num_objects[b]:
            ncandidates *= 1.5
            #logger.info(f"only sampled {pre_positions.shape[0]} objects, increased ncandidate - now in {ncandidates}")
            engine = qmc.PoissonDisk(d=2, radius=sampling_radius, ncandidates=int(ncandidates))
            pre_positions = engine.random(num_objects[b])
       
        pre_positions[:, 0] = pre_positions[:, 0] * (x_max - x_min) + x_min  # Scale X positions
        pre_positions[:, 1] = pre_positions[:, 1] * (y_max - y_min) + y_min  # Scale Y positions
        pre_positions = torch.tensor(pre_positions)
        
        all_positions = torch.zeros(M, 2)
        real_obj_idx = torch.nonzero(objects_mask[b])
        if real_obj_idx.shape[0] != 0:
            real_obj_idx = real_obj_idx[:, 0]
            s_idx = 0
            for idx in real_obj_idx:
                all_positions[idx] = pre_positions[s_idx]
                s_idx += 1
        positions.append(all_positions)
    positions = torch.stack(positions) # [B, M, 2]

    #with pyro.poutine.mask(mask=objects_mask):
    with pyro.poutine.block():
        x = pyro.sample(f"x", dist.Normal(positions[:, :, 0]/3., llh_uncertainty))*3.
        y = pyro.sample(f"y", dist.Normal(positions[:, :, 1]/3., llh_uncertainty))*3.

    with pyro.poutine.mask(mask=objects_mask):
        
        if params['running_type'] == 'eval': 
            if x.dim() > 2:
                x = torch.flatten(x, 0, 1)
                y = torch.flatten(y, 0, 1)
        
        if params['running_type'] == 'eval': 
            if size.dim() > 2:
                size = torch.flatten(size, 0, 1)        

    # For 'Cube', adjust 'r'
    for b in range(B):
        for k, name in enumerate(obj_name[b]):
            if name == 'Cube':
                r[b][k] /= math.sqrt(2)

    # Store each scene's attributes
    scenes = []
    
    """ sample the '3d_coords' tensor according to x, y and size """

    if params["running_type"] == "train":
        x = (x + 3.)/6.
        y = (y + 3.)/6.
   
    with pyro.poutine.mask(mask=objects_mask):
        
        if params["running_type"] == "train":
            size_to_z = {
                0: torch.tensor(0.70),
                1: torch.tensor(0.35)
            }
            z = torch.stack([size_to_z[s.item()] for s in size.flatten()]).view(B, M)
            z = (z + 3.)/6.
        else:
            with pyro.poutine.block():
                z = pyro.sample("z", dist.Uniform(-3., 3.)).expand([B, M])
        
        coords = pyro.sample(f"coords", dist.Normal(torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=-1),
                                                    llh_uncertainty*0.1))
        if params["running_type"] == "eval":
            #logger.info(f"pre coords: {coords}")
            coords = coords * 6 - 3.
            #logger.info(f"post coords: {coords}")
    
    for b in range(B):
        objects = []
        # Append the object attributes to the scene list
        for k in range(M):
            if objects_mask[b, k]:
                objects.append({
                    "shape": obj_name[b][k],
                    "color": color_name[b][k],
                    "rgba": rgba[b][k],
                    "size": r[b][k],
                    "material": mat_name[b][k],
                    "pose": theta[b, k].item(),
                    "position": (coords[b, k, 0].item(), coords[b, k, 1].item())
                })
        
        scenes.append(objects)

    return scenes

def generate_blender_script(objects, id, save_dir, gen_samples):
    """
    Generate a Blender Python script to render the CLEVR-like scene.
    """
    script = f"""
import bpy
import random
import os
import logging

from mathutils import Vector

logger = logging.getLogger('blender_logger')
logger.setLevel(logging.INFO)  # You can change this level to INFO, WARNING, etc.
log_file = os.path.join(os.path.dirname(__file__), 'blender_log.log')
handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#logger.info('logging from the generated blender script!')

# Set directory path
main_path = os.path.join("/nas-ctm01", "homes", "fcsilva", "ic-slotatt", "main")

#logger.info(main_path)

# Set images and blender files path
imgs_path = r"{save_dir}"

#logger.info(imgs_path)

# Open main file
bpy.ops.wm.open_mainfile(filepath=os.path.join(main_path, "clevr_data", "base_scene.blend"))

# Set render arguments so we can get pixel coordinates later.
# We use functionality specific to the CYCLES renderer so BLENDER_RENDER
# cannot be used.
render_args = bpy.context.scene.render
render_args.engine = "CYCLES"
render_args.resolution_x = 320
render_args.resolution_y = 240
render_args.resolution_percentage = 100

# Some CYCLES-specific stuff
bpy.data.worlds['World'].cycles.sample_as_light = True
bpy.context.scene.cycles.blur_glossy = 2.0
bpy.context.scene.cycles.samples = {gen_samples}
bpy.context.scene.cycles.transparent_min_bounces = 8
bpy.context.scene.cycles.transparent_max_bounces = 8
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

bpy.context.preferences.addons['cycles'].preferences.get_devices()
devices = bpy.context.preferences.addons['cycles'].preferences.devices
for device in devices:
    device.use = True

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

load_materials(os.path.join(main_path, "clevr_data", "materials"))

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
    
    shape_dir = os.path.join(main_path, "clevr_data", "shapes")
    add_object(shape_dir, object_dir["shape"], object_dir["size"], object_dir["position"], object_dir["pose"])
    
    # Get reference to the object
    obj = bpy.context.object
    
    # Add material for the object
    add_material(object_dir['material'], Color=object_dir['rgba'])


# Sampled objects from Pyro

#logger.info("adding objects to blender scene...")
"""
    script += """

objects = {}
"""
    
    script += f"""

# Pass the index of the batched sample
idx = {id}
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
bpy.context.scene.render.filepath = os.path.join(imgs_path, f"rendered_scene_{idx}.png")

#logger.info(os.path.join(imgs_path, f"rendered_scene_{idx}.png"))

# Render the scene
bpy.ops.render.render(write_still=True)
    """
    
    # Write the Blender script to a file
    script_file = os.path.join(save_dir, f"generate_clevr_scene_{id}.py")
    with open(script_file, "w") as f:
        f.write(script)
    
    return script_file

def render_scene_in_blender(blender_script):
    """
    Call Blender to execute the script and render the scene.
    """
    debug_log = os.path.join(dir_path, "debug_output.log")

    blender_path = "/usr/bin/blender"  # Update this with your Blender path
    cmd = [blender_path, "--background", "--python", blender_script, "-d"]

    with open(debug_log, "w") as log_file:
        subprocess.call(cmd, stdout=log_file, stderr=log_file)



def clevr_gen_model(step=0, observations={"image": torch.zeros((1, 3, 128, 128))}, pixel_coords=None):

    if params['running_type'] == 'train': llh_uncertainty = 0.001
    elif params['running_type'] == 'eval': llh_uncertainty = 0.05

    if params['running_type'] == "train":
        if not os.path.isdir(os.path.join(dir_path, str(params['jobID']))): os.mkdir(os.path.join(dir_path, str(params['jobID'])))
        if not os.path.isdir(os.path.join(dir_path, str(params['jobID']), "train")): os.mkdir(os.path.join(dir_path, str(params['jobID']), "train"))
        imgs_path = os.path.join(dir_path, str(params['jobID']), "train")
    elif params['running_type'] == "eval":
        if not os.path.isdir(os.path.join(dir_path, str(params['jobID']))): os.mkdir(os.path.join(dir_path, str(params['jobID'])))
        assert os.path.isdir(os.path.join(dir_path, str(params['jobID'])))
                
        if not os.path.isdir(os.path.join(dir_path, str(params['jobID']), "eval")): os.mkdir(os.path.join(dir_path, str(params['jobID']), "eval"))
        if not os.path.isdir(os.path.join(dir_path, str(params['jobID']), "eval", f"split_{JOB_SPLIT['id']}")): 
            os.mkdir(os.path.join(dir_path, str(params['jobID']), "eval", f"split_{JOB_SPLIT['id']}"))
        imgs_path = os.path.join(dir_path, str(params['jobID']), "eval", f"split_{JOB_SPLIT['id']}")

    #logger.info(imgs_path)

    # delete all blender scripts
    files = glob.glob(os.path.join(imgs_path, "*.py"))
    for f in files:
        if f.split('/')[-1].split('_')[:3] == ["generate", "clevr", "scene"]: os.remove(f)
    
    # delete all generated imgs
    imgs = glob.glob(os.path.join(imgs_path, "*.png"))
    for img in imgs:
        if img.split('/')[-1].split('_')[:2] == ["rendered", "scene"]: os.remove(img)

    
    #init_time = time.time()
    # Sample a CLEVR-like scene using Pyro
    clevr_scenes = sample_clevr_scene(llh_uncertainty, pixel_coords)
    #sample_time = time.time() - init_time
    #if params["running_type"] == "train": logger.info(f"Scene sampling time: {sample_time}")

    B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']

    #init_time = time.time()
    # Generate the Blender script for the sampled scene
    if params["running_type"] == "train":
        gen_samples = 128 if torch.distributions.Bernoulli(0.8).sample() else 512
    else:
        gen_samples = 512 
    blender_scripts = [generate_blender_script(scene, idx, imgs_path, gen_samples) for idx, scene in enumerate(clevr_scenes)]
    #script_time = time.time() - init_time
    #if params["running_type"] == "train": logger.info(f"Scene scripting time: {script_time}")
    
    #logger.info(os.listdir(imgs_path))

    # Call Blender to render the scene
    #with mp.Pool(processes=mp.cpu_count()) as pool:
    init_time = time.time()
    with mp.Pool(processes=10) as pool:
      pool.map(render_scene_in_blender, blender_scripts)
    batch_time = time.time() - init_time
    if params["running_type"] == "train" and step % params["step_size"]: logger.info(f"Batch generation duration: {batch_time} - {batch_time/B} per sample")

    # init_time = time.time()
    # for blender_script in blender_scripts:
    #     render_scene_in_blender(blender_script)
    # batch_time = time.time() - init_time
    # logger.info(f"Batch generation duration: {batch_time} - {batch_time/B} per sample")
    
    #logger.info(os.listdir(imgs_path))

    #logger.info("Scene rendered and saved...")
    # img_batch = torch.stack(
    #     [torch.from_numpy(np.asarray(Image.open(os.path.join(imgs_path, f"rendered_scene_{idx}.png")).convert('RGB'))).permute(2, 0, 1) for idx in range(B)]
    # )
    img_batch = torch.stack([transform(Image.open(os.path.join(imgs_path, f"rendered_scene_{idx}.png")).convert('RGB')) for idx in range(B)])

    #logger.info(img_batch.shape)

    img_batch = img_batch*2 - 1

    #proc_img = preprocess_clevr(img_batch) # proc_img shape is (1, 4, 128, 128)

    #logger.info(f"gen image shape: {proc_img.shape}")

    #with pyro.plate(observations["image"].shape[0]):
        #pyro.sample("image", MyBernoulli(proc_img, validate_args=False).to_event(3), obs=observations["image"])
        
        # stddev = 0.01  in jobID 78
        # stddev = 0.001 in jobID 79
    
    likelihood_fn = MyNormal(img_batch, torch.tensor(llh_uncertainty)).get_dist() 
    pyro.sample("image", likelihood_fn.to_event(3), obs=observations["image"])
    
    
    
    