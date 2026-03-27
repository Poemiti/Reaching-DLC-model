from pathlib import Path
import yaml
import json
from reaching_model_utils.video_utils import extract_frames_uniform, extract_frames_phash
import reaching_model_utils.evaluation_utils as evaluation_utils
from reaching_model_utils.config import load_config
import sys
import deeplabcut

# ----------------------------------- setup path and parameters -------------------------------------

cfg = load_config("../config.yaml")

root = "/media/filer2/T4b/UserFolders/Poemiti"
config_path = evaluation_utils.select_file(root, title="Select Config.yaml of the model to evalutate",
                                          filetype=[("YAML files", "*.yaml")])

deeplabcut.check_labels(config_path)

