
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml, os, json, cv2, csv, timeit
import deeplabcut
import shutil
shutil.copy = shutil.copyfile

import utils


# ------------------------------ verify gpu ------------------------------------

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# ----------------------------- setup parameters -----------------------------

# training params
optimizer = "AdamW"
n_epochs = 300
batch_size = 12
num_frames_for_train = 400 


# Names
project = "DLC"
experimenter = "Poe"


# directories path
root_dir = Path(".") 
images_dir = root_dir / "Labeling/Images"
annotation_dir = root_dir / "Labeling/Annotations"


# files path
info_skeleton = Path("./info_skeleton.yaml") 
json_list_path = root_dir / "annotations_list.json"
temp_video_path = root_dir/ "Temporary/output_video.avi"



# --------------------------- Annotation ----------------------------

print("\nLoafing annotation\n")

all_annotations, frame_map = [], {}

txt_files = sorted(os.listdir(annotation_dir))[:num_frames_for_train]
for i, fname in enumerate(txt_files):
    with open(annotation_dir / fname, "r", encoding="utf-8") as f:
        annot = json.load(f)
    all_annotations.append(annot)
    rel_img = annot.get('task', {}).get('data', {}).get('rel_img_path')
    if rel_img:
        frame_name = os.path.basename(rel_img)
        frame_map[str(i)] = frame_name
        full_img = images_dir / rel_img
        target_img = images_dir / frame_name
        if not target_img.exists() and full_img.exists():
            shutil.copy(full_img, target_img)

with open(json_list_path, 'w', encoding='utf-8') as f:
    json.dump(all_annotations, f, indent=4, ensure_ascii=False)



# --------------------------- Create videos from Labeling/Images ----------------------------

print("\nCreating video from Labeling/Images\n")

utils.create_video_from_frames(input_dir=images_dir, 
                               output_video_path=temp_video_path, 
                               fps=1, 
                               num=num_frames_for_train, 
                               frame_map=frame_map)


# --------------------------- Create DLC project ----------------------------

print("\nCreating DLC project\n")
deeplabcut.create_new_project(project=project, 
                              experimenter=experimenter, 
                              videos = [str(temp_video_path)], 
                              working_directory = root_dir, 
                              copy_videos=True, multianimal=False)


# --------------------------- update config ----------------------------

print("\nUpdate config.yaml\n")

date_str = datetime.today().strftime('%Y-%m-%d')
config_path = root_dir / f"{project}-{experimenter}-{date_str}" / "config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 1. training params
config.update({
    'numframes2extract': num_frames_for_train,
    'numframes2pick': num_frames_for_train,
    'pcutoff': 0.0,
    'dotsize': 5,
    'batch_size': batch_size,
    'engine': 'pytorch'
})

# 2. dataset structure (after annotation processing)
with open(info_skeleton, "r") as f:
    info_bs = yaml.safe_load(f)

original_skeleton = info_bs.get('skeleton', [])
annotated_bodyparts = {
    result.get('value', {}).get('keypointlabels', [])[0]
    for annot in all_annotations
    for result in annot.get('result', [])
    if result.get('type') == 'keypointlabels'
}

config['bodyparts'] = sorted(annotated_bodyparts)
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
dlc_frames_dir = root_dir / f"{project}-{experimenter}-{date_str}" / "labeled-data" / "output_video"
utils.extract_exact_frames_from_video(temp_video_path, dlc_frames_dir, frame_map)

# Conversion des annotations en .csv
csv_path = root_dir / f"{project}-{experimenter}-{date_str}" / "output.csv"

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['labeller', 'video_reference', 'frame_id', 'bodypart', 'x', 'y', 'confidence'])
    
    for annotation in all_annotations:
        task_data = annotation.get('task', {}).get('data', {})
        video_ref = os.path.basename(os.path.dirname(task_data.get('source_video_filepath', '')))
        frame_id = os.path.basename(task_data.get('rel_img_path', f"{task_data.get('frame_num', 0)}.png"))
        
        for result in annotation.get('result', []):
            if result.get('type') != 'keypointlabels':
                continue
            
            value = result.get('value', {})
            x = value.get('x', np.nan)
            y = value.get('y', np.nan)
            ow = result.get('original_width', 1)
            oh = result.get('original_height', 1)
            x, y = (x / 100 * ow), (y / 100 * oh)
            bp = value.get('keypointlabels', [''])[0]
            conf = 1 if not np.isnan(x) and not np.isnan(y) else np.nan
            writer.writerow([experimenter, video_ref, frame_id, bp, x, y, conf])

# Conversion .csv to .h5
h5 = utils.csv_to_h5(str(csv_path))
h5.to_hdf(str(dlc_frames_dir / f"CollectedData_{experimenter}.h5"), "keypoints")


# -------------------------------- creating training dataset ----------------------------------

print("\nCreating training dataset\n")

deeplabcut.create_training_dataset(config = str(config_path))


# -------------------------------- update pytorch config file ----------------------------------

print("\nUpdating pytorch_config.yaml\n")

# Modification de pytorch_config.yaml
month_day = datetime.today().strftime("%b") + str(datetime.today().day)
pytorch_config_path = root_dir / f"{project}-{experimenter}-{date_str}" / "dlc-models-pytorch" / "iteration-0" / f"DLC{month_day}-trainset95shuffle1" / "train" / "pytorch_config.yaml"

with open(pytorch_config_path, 'r') as f:
    config_py = yaml.safe_load(f)
config_py['runner']['optimizer']['type'] = optimizer
config_py['train_settings']['epochs'] = n_epochs
config_py['snapshot'] = 50
config_py['train_settings']['batch_size'] = batch_size

with open(pytorch_config_path, 'w') as f:
    yaml.safe_dump(config_py, f)



# -------------------------------- Training ----------------------------------

print("\nTRAINING HAS STARTED\n")


start = timeit.default_timer()
deeplabcut.train_network(str(config_path), shuffle=1)
stop = timeit.default_timer()
print(f"Training time = {stop - start:.2f} sec")



