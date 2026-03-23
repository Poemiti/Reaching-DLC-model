from typing import List, Dict, Literal, Optional, Any, ClassVar
from pathlib import Path
import yaml
import json
from src.reaching_model_utils.video_utils import extract_frames_uniform, extract_frames_phash



# ----------------------------------- setup path and parameters -------------------------------------

# parameters
labeling_project_name = "Model_Poe"
random_state = 42
num_frames_per_video = 25
extract_method = "phash"
videos = [""]


# existing directories
root = Path("./data/")
labelling_dir = root / "data/labeling/"


# output directories
json_output_path = labelling_dir / 'frames_metadata.json'
frames_output_dir = labelling_dir / "Images"


# ----------------------------------- frame extraction -------------------------------------

with (labelling_dir / "project_info.yaml").open("r") as f:
    metadata = yaml.safe_load(f)

if extract_method == 'uniform':
    for video in videos:
        json_data = extract_frames_uniform(video, frames_output_dir, num_frames_per_video, metadata, 
                             labeling_project_name=labeling_project_name)



elif extract_method == 'phash':
    for video in videos:

        json_data = extract_frames_phash(video, frames_output_dir, 
                             max_frames=num_frames_per_video, 
                             step=10, phash_threshold=10, 
                             metadata=metadata, 
                             labeling_project_name=labeling_project_name)
        

else:
    raise Exception("Only 'uniform' and 'phash' method are implemented.")


# save 
with open(json_output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)