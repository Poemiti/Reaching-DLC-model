from pathlib import Path
import yaml
import json
from reaching_model_utils.config import load_config
import sys
import os
from collections import Counter

# ----------------------------------- setup path and parameters -------------------------------------

cfg = load_config("../config.yaml")

annotation_path = cfg.paths.labeling / "frames_annotations_meta.json"
bodyparts_path = Path("../info_skeleton_V3.yaml")
exclusion_path = Path("../annotations_exclusion_list.yaml")

with open(exclusion_path, "r") as f : 
    exclusion_cfg = yaml.safe_load(f)

with open(bodyparts_path, "r") as f : 
    bodyparts_cfg = yaml.safe_load(f)

with open(annotation_path, "r") as f : 
    annotations = json.load(f)

expected_bodyparts = bodyparts_cfg["bodyparts"]
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

    excluded = annot_id in exclusion_cfg

    if missing or duplicates or excluded:
        wrong_annotations[annot_id] = {
            "img": img_path,
            "bodyparts": bodyparts,
            "missing": list(missing),
            "duplicates": duplicates, 
            "excluded": excluded
        }


if len(wrong_annotations) < 0 : 
    print("\nNO wrong annotation found !\n")
    sys.exit()

for k, v in wrong_annotations.items():
    if v["excluded"] == True: 
        continue

    print(f"\nAnnotation {k} : ")
    print(f"  img: {v['img']}")
    if len(v["missing"]) > 0 : 
        print(f"  Missing: {v['missing']}")
    if len(v["duplicates"]) > 0 : 
        print(f"  Duplicates: {v['duplicates']}")

n_wrong_annot = sum(v["excluded"] == False for v in wrong_annotations.values())
n_excluded_annot = len(wrong_annotations) - n_wrong_annot 
print(f"\n[{n_wrong_annot}/{len(annotations)}] wrong annotation were found !")
print(f"[{n_excluded_annot}/{len(wrong_annotations)}] annotation were excluded !\n")
