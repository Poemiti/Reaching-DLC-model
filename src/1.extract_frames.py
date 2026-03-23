from pathlib import Path
import yaml
import json
from reaching_model_utils.video_utils import extract_frames_uniform, extract_frames_phash
from reaching_model_utils.config import load_config



# ----------------------------------- setup path and parameters -------------------------------------

cfg = load_config()

json_output_path = cfg.paths.labeling / 'frames_metadata.json'
frames_output_dir = cfg.paths.labeling  / "Images"
annotation_output_dir = cfg.paths.labeling  / "Annotations"

frames_output_dir.mkdir(parents=True, exist_ok=True)
annotation_output_dir.mkdir(parents=True, exist_ok=True)

videos = [
    Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024/#525/25062024/Rat_#525Ambidexter_20240625_BetaMT300_RightHemiCHR_L1L25050_C001H002S0002/Rat_#525Ambidexter_20240625_BetaMT300_RightHemiCHR_L1L25050_C001H002S0002.avi"),
    ]


# ----------------------------------- frame extraction -------------------------------------

with (cfg.paths.labeling  / "project_info.yaml").open("r") as f:
    metadata = yaml.safe_load(f)

if cfg.extract_method == 'uniform':
    for video in videos:
        print(f"\nUniform extraction of : {video.stem}")
        json_data = extract_frames_uniform(video, frames_output_dir, cfg.num_frames_per_video, metadata, 
                             labeling_dir=cfg.paths.labeling )



elif cfg.extract_method == 'phash':
    for video in videos:
        print(f"\nPhash extraction of : {video.stem}")
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