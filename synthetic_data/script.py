import json 
import os
import glob

import logging
logfile_name = "generate.log"
logging.basicConfig(filename=logfile_name, level=logging.INFO)


NSCENES = 30000

# path for local generation
#metadata_dir = os.path.abspath('color-metadata')
#img_dir = os.path.abspath('color-images')
#dot_annot_dir = os.path.abspath('color-annots')

# path for slurm generation
metadata_dir = '' # path for dataset metadata
max_objects = 0

for scene_id in range(NSCENES):
  logging.info(scene_id)
  with open(f"{metadata_dir}/{str(scene_id).zfill(5)}.json", "r") as outfile:
    scene_dict = json.load(outfile)  
    if scene_dict['scene_attr']['N'] > max_objects: max_objects = scene_dict['scene_attr']['N']

logging.info(f"all dicts analysed: max # of objects is {max_objects}...")


