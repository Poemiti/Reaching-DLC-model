
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml, os, json, cv2, csv, timeit
import shutil
shutil.copy = shutil.copyfile

import reaching_model_utils.video_utils as video_utils
from reaching_model_utils.config import load_config
import reaching_model_utils.training_utils as training_utils

# ------------------------------ verify gpu ------------------------------------

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# ----------------------------- setup parameters -----------------------------

cfg = load_config("../config.yaml")

date_str = datetime.today().strftime('%Y-%m-%d')
project_name = f"{cfg.project}-{cfg.experimenter}-{date_str}"

images_dir = cfg.paths.labeling / "Images"
annotation_dir = cfg.paths.labeling / "Annotations"
dlc_frames_dir = cfg.paths.model / project_name / "labeled-data" / "output_video"



# files path
info_skeleton = Path("../info_skeleton.yaml") 
json_list_path = cfg.paths.model / "annotations_list.json"
temp_video_path = cfg.paths.temporary / "output_video.avi"
annotation_path = cfg.paths.labeling / "frames_annotations_meta.json"
csv_path = cfg.paths.model / project_name / "output.csv"
config_path = cfg.paths.model / project_name / "config.yaml"


single_file_annotation = len(os.listdir(annotation_dir)) != 0


# --------------------------- Annotation ----------------------------

print("\nLoading annotation\n")

if single_file_annotation : 
    print("Annotation directory is NOT empty")
    all_annotations = []
    txt_files = sorted(os.listdir(annotation_dir))[:cfg.num_frames_for_train]

    for i, fname in enumerate(txt_files):
        with open(annotation_dir / fname, "r", encoding="utf-8") as f:
            annot = json.load(f)
        all_annotations.append(annot)

frame_map = training_utils.build_frame_map(all_annotations, images_dir, single_file_annotation)

with open(json_list_path, 'w', encoding='utf-8') as f:
    json.dump(all_annotations, f, indent=4, ensure_ascii=False)

# --------------------------- Create videos from Labeling/Images ----------------------------


if len(os.listdir(temp_video_path.parent)) == 0 : 
    print("\nCreating video from Labeling/Images\n")
    video_utils.create_video_from_frames(input_dir=images_dir, 
                                output_video_path=temp_video_path, 
                                fps=1, 
                                num=cfg.num_frames_for_train, 
                                frame_map=frame_map)


# --------------------------- Create DLC project ----------------------------

import deeplabcut

print("\nCreating DLC project\n")
deeplabcut.create_new_project(project=cfg.project, 
                              experimenter=cfg.experimenter, 
                              videos = [str(temp_video_path)], 
                              working_directory = cfg.paths.model, 
                              copy_videos=True, multianimal=False)


# --------------------------- update config ----------------------------

print("\nUpdate config.yaml\n")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 1. training params
config.update({
    'numframes2extract': cfg.num_frames_for_train,
    'numframes2pick': cfg.num_frames_for_train,
    'pcutoff': 0.0,
    'dotsize': 5,
    'batch_size': cfg.batch_size,
    'engine': 'pytorch'
})

# 2. dataset structure (after annotation processing)
with open(info_skeleton, "r") as f:
    info_bs = yaml.safe_load(f)

original_bodyparts = info_bs.get('bodyparts', [])
original_skeleton = info_bs.get('skeleton', [])
annotated_bodyparts = training_utils.extract_bodyparts(all_annotations, single_file_annotation)

config['bodyparts'] = original_bodyparts
config['skeleton'] = [
    link for link in original_skeleton
    if isinstance(link, list)
    and len(link) == 2
    and link[0] in annotated_bodyparts
    and link[1] in annotated_bodyparts
]

# save update
with open(config_path, 'w') as f:
    yaml.safe_dump(config, f)


# --------------------------- convert annotation for DLC ----------------------------

print("\nExtracting frames and converting annotation\n")

# Extraction des images
if single_file_annotation : 
    video_utils.extract_exact_frames_from_video(temp_video_path, dlc_frames_dir, frame_map)
else : 
    video_utils.copy_frames_to_dir(images_dir, dlc_frames_dir)

# Conversion des annotations en .csv
training_utils.write_dlc_csv(csv_path, all_annotations, cfg,
                             original_bodyparts, single_file_annotation)

# Conversion .csv to .h5
h5 = video_utils.csv_to_h5(str(csv_path))
h5.to_hdf(str(dlc_frames_dir / f"CollectedData_{cfg.experimenter}.h5"), "keypoints")


# -------------------------------- creating training dataset ----------------------------------

print("\nCreating training dataset\n")

deeplabcut.create_training_dataset(config = str(config_path))


# -------------------------------- update pytorch config file ----------------------------------

print("\nUpdating pytorch_config.yaml\n")

# Modification de pytorch_config.yaml
month_day = datetime.today().strftime("%b") + str(datetime.today().day)
pytorch_config_path = cfg.paths.model / project_name / "dlc-models-pytorch" / "iteration-0" / f"DLC{month_day}-trainset95shuffle1" / "train" / "pytorch_config.yaml"

with open(pytorch_config_path, 'r') as f:
    config_py = yaml.safe_load(f)
config_py['runner']['optimizer']['type'] = cfg.optimizer
config_py['train_settings']['epochs'] = cfg.n_epochs
config_py['snapshot'] = 50
config_py['runner']['eval_interval'] = 1
config_py['train_settings']['batch_size'] = cfg.batch_size

with open(pytorch_config_path, 'w') as f:
    yaml.safe_dump(config_py, f)



# -------------------------------- Training ----------------------------------

print("\nTRAINING HAS STARTED\n")


start = timeit.default_timer()
deeplabcut.train_network(str(config_path), shuffle=1)
stop = timeit.default_timer()
print(f"Training time = {stop - start:.2f} sec")



# -------------------------------- Evaluation ----------------------------------


deeplabcut.evaluate_network(config_path, Shuffles=[1], 
                            plotting=True,
                            per_keypoint_evaluation=True)
