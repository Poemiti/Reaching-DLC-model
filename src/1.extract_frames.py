from pathlib import Path
import yaml
import json
from reaching_model_utils.video_utils import extract_frames_uniform, extract_frames_phash
from reaching_model_utils.config import load_config
from tqdm import tqdm
import sys
import os


# ----------------------------------- setup path and parameters -------------------------------------

cfg = load_config()

json_output_path = cfg.paths.labeling / 'frames_metadata.json'
frames_output_dir = cfg.paths.labeling  / "Images"
annotation_output_dir = cfg.paths.labeling  / "Annotations"

frames_output_dir.mkdir(parents=True, exist_ok=True)
annotation_output_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------- verification step -------------------------------------

# verification videos
for v in cfg.video_to_extract : 
    v.exists()

if len(os.listdir(frames_output_dir)) != 0 : 
    res = input("The output folder is not empty, do you want to overwrite ? (y/n) : ")
    if res == "n" : 
        print("Extraction cancelled, stop!")
        sys.exit()

print("================================================")
print(f"Output folder: {str(frames_output_dir)}")
print(f"Number of video: {len(cfg.video_to_extract)}")
print(f"Number of frame that will be extracted: {len(cfg.video_to_extract)*cfg.num_frames_per_video}")
print(f"Extracting method used: {cfg.extract_method}")
print("================================================")


# # ----------------------------------- frame extraction -------------------------------------

json_data = []

if cfg.extract_method == 'uniform':
    for video in cfg.video_to_extract:
        print(f"\nUniform extraction of : {video.stem}")
        data = extract_frames_uniform(video, frames_output_dir, cfg.num_frames_per_video, 
                             labeling_dir=cfg.paths.labeling )
        json_data.extend(data)



elif cfg.extract_method == 'phash':
    for video in cfg.video_to_extract:
        print(f"\nPhash extraction of : {video.stem}")
        data = extract_frames_phash(video, frames_output_dir, 
                             max_frames=cfg.num_frames_per_video, 
                             step=10, phash_threshold=10, 
                             labeling_dir=cfg.paths.labeling )
        json_data.extend(data)
        

else:
    raise Exception("Only 'uniform' and 'phash' method are implemented.")


# save 
with open(json_output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print("Done !")