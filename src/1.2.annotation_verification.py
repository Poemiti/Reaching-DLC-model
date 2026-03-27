from pathlib import Path
import yaml
import json
from reaching_model_utils.video_utils import extract_frames_uniform, extract_frames_phash
from reaching_model_utils.config import load_config
from tqdm import tqdm
import sys
import os
import pandas as pd
from collections import Counter

# ----------------------------------- setup path and parameters -------------------------------------

cfg = load_config("../config.yaml")

annotation_path = cfg.paths.labeling / "frames_annotations_meta.json"


with open(annotation_path, "r") as f : 
    annotations = json.load(f)

expected_bodyparts = ["elbow", "finger_1", "finger_2", "finger_3", 
                      "hand", "muzzle", "shoulder", "soft_pad", "wrist"]
wrong_annotations = {}

for annot in annotations : 
    annot_id = annot["id"]
    img_path = os.path.basename(annot["data"]["rel_img_path"])
    bodyparts = [
        label
        for result in annot["annotations"][0].get('result', [])
        if result.get('type') == 'keypointlabels'
        for label in result.get('value', {}).get('keypointlabels', [])
    ]

    expected_set = set(expected_bodyparts)
    found_set = set(bodyparts)

    missing = expected_set - found_set

    counts = Counter(bodyparts)
    duplicates = {k: v for k, v in counts.items() if v > 1}

    if missing or duplicates:
        wrong_annotations[annot_id] = {
            "img": img_path,
            "bodyparts": bodyparts,
            "missing": list(missing),
            "duplicates": duplicates
        }


if len(wrong_annotations) < 0 : 
    print("\nNO wrong annotation found !\n")
    sys.exit()

print(f"\n{len(wrong_annotations)} wrong annotation were found ! :")

for k, v in wrong_annotations.items():
    print(f"\nAnnotation {k} : ")
    print(f"  img: {v['img']}")
    if len(v["missing"]) > 0 : 
        print(f"  Missing: {v['missing']}")
    if len(v["duplicates"]) > 0 : 
        print(f"  Duplicates: {v['duplicates']}")