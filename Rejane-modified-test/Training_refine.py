
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
n_epochs = 500
batch_size = 8
num_frames_for_train = 200 


# Names
project = "DLC"
experimenter = "Poe"
trained_model_experimenter = "project"


# directories path
root_dir = Path(".") 
images_dir = root_dir / "Labeling/Images"
annotation_dir = root_dir / "Labeling/Annotations"
trained_model_dir = Path("/media/filer2/T4b/Models/DLC/REJANE_rat_right_model-2025-06-18/Modelconfig_predict_24_200_1000/DLC-project-2025-06-17")

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

date_str = datetime.today().strftime('%Y-%m-%d')
new_config_path = root_dir / f"{project}-{experimenter}-{date_str}" / "config.yaml"
trained_model_config_path = trained_model_dir / "config.yaml"

print("\nAdding new videos\n")
deeplabcut.add_new_videos(config=trained_model_config_path,
                          videos=temp_video_path)

# --------------------------- update config ----------------------------

print("\nUpdate config.yaml\n")

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