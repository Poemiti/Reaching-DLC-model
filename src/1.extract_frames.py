from pathlib import Path
import yaml
import json
from src.reaching_model_utils.video_utils import extract_frames_uniform, extract_frames_phash
from src.reaching_model_utils.config import load_config



# ----------------------------------- setup path and parameters -------------------------------------

cfg = load_config()

json_output_path = cfg.paths.labeling / 'frames_metadata.json'
frames_output_dir = cfg.paths.labeling  / "Images"


videos = [""]


# ----------------------------------- frame extraction -------------------------------------

with (cfg.paths.labeling  / "project_info.yaml").open("r") as f:
    metadata = yaml.safe_load(f)

if cfg.extract_method == 'uniform':
    for video in videos:
        json_data = extract_frames_uniform(video, frames_output_dir, cfg.num_frames_per_video, metadata, 
                             labeling_dir=cfg.paths.labeling )



elif cfg.extract_method == 'phash':
    for video in videos:

        json_data = extract_frames_phash(video, frames_output_dir, 
                             max_frames=cfg.num_frames_per_video, 
                             step=10, phash_threshold=10, 
                             metadata=metadata, 
                             labeling_dir=cfg.paths.labeling )
        

else:
    raise Exception("Only 'uniform' and 'phash' method are implemented.")


# save 
with open(json_output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)