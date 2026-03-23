
import yaml, os, json, cv2, csv, timeit
from typing import List, Dict, Literal, Optional, Any, ClassVar

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import imagehash
from PIL import Image
import cv2





# ----------------------------------- for frame extraction -------------------------------------


# uniform method
def extract_frames_uniform(video_path: Path, output_dir, num_frames: int, metadata: Dict[str, Any], labeling_project_name: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames / num_frames
    global_id_counter = 0
    json_data = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read() 
        if not ret:
            break

        parent_dict = {f"parent_{j+1}": p.name for j, p in enumerate(video_path.parents)}
        metadata_filled = {key: (value.format(frame_num=i+1, **parent_dict) if isinstance(value, str) else value) for key, value in metadata.items()}
        filename_parts = [video_path.stem, f"uniform_frame_{i+1}"]
        filename_parts += [f"{value}" for value in metadata_filled.values()]
        new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + "_" + f"img{i+1}.png"
        frame_filename = output_dir / new_frame_name

        counter = 1
        while frame_filename.exists():
            new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + "_" + f"img{i+1}{counter}.png"
            frame_filename_new = Path(f"{labeling_project_name}/Images/{new_frame_name}")
            frame_filename = output_dir / new_frame_name
            counter += 1

        cv2.imwrite(str(frame_filename), frame)
        label_studio_base_path = 'http://10.24.12.180:8083/ls/'
        frame_data = {
            "id": global_id_counter + 1,
            "data": {
                "frame_num": f"{i+1}", 
                "rel_img_path": str(frame_filename.relative_to(output_dir.parent)).replace('#', '%23'),
                "label_studio_img_path": f"{label_studio_base_path}{str(frame_filename_new).replace('#', '%23')}",
                "source_video_filepath": str(video_path.resolve()),
            }}
        frame_data["data"].update(metadata_filled)
        json_data.append(frame_data)
        global_id_counter += 1  
    cap.release()


    return json_data



# phash method
def phash_distance(frame1, frame2):
    pil1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    h1 = imagehash.phash(pil1)
    h2 = imagehash.phash(pil2)
    return h1 - h2



def extract_frames_phash(video_path: Path, 
                         output_dir: Path, 
                         max_frames: int, 
                         step: int, 
                         phash_threshold: int, 
                         metadata: Dict[str, Any], 
                         labeling_project_name: str):
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Problem for : {video_path}")

    selected = []
    prev_frame = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    global_id_counter = 0
    json_data = []

    while len(selected) < max_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if prev_frame is None or phash_distance(prev_frame, frame) > phash_threshold:
            i = len(selected)
            parent_dict = {f"parent_{j+1}": p.name for j, p in enumerate(video_path.parents)}
            metadata_filled = {key: (value.format(frame_num=i+1, **parent_dict) if isinstance(value, str) else value) for key, value in metadata.items()}
            new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + f"_img{i+1}.png"
            frame_filename = output_dir / new_frame_name
            counter = 1
            while frame_filename.exists():
                new_frame_name = f"frame" + "_".join(map(str, metadata_filled.values())) + f"_img{i+1}_{counter}.png"
                frame_filename = output_dir / new_frame_name
                counter += 1
            
            print(frame_filename)
            cv2.imwrite(str(frame_filename), frame)
            label_studio_base_path = 'http://10.24.12.180:8083/ls/'
            frame_data = {
                "id": global_id_counter + 1,
                "data": {
                    "frame_num": f"{i+1}",
                    "rel_img_path": str(frame_filename.relative_to(output_dir.parent)).replace('#', '%23'),
                    "label_studio_img_path": f"{label_studio_base_path}{labeling_project_name}/Images/{new_frame_name}",
                    "source_video_filepath": str(video_path.resolve())
                    }}
            frame_data["data"].update(metadata_filled)
            json_data.append(frame_data)
            selected.append(frame)
            prev_frame = frame
            global_id_counter += 1
        idx += step
    cap.release()

    return json_data




# ----------------------------------- for training -------------------------------------



def create_video_from_frames(input_dir: Path, 
                             output_video_path: Path, 
                             fps: int = 1, 
                             num: int = None, 
                             frame_map: dict = None) -> None:
    
    input_dir = Path(input_dir)
    # ordered_frame_names = [frame_map[str(i)] for i in range(min(len(frame_map), num or len(frame_map)))]
    
    if frame_map:
        ordered_keys = sorted(frame_map.keys(), key=lambda x: int(Path(x).stem))
        ordered_frame_names = [frame_map[k] for k in ordered_keys[:num]]

    else:
        all_images = sorted(input_dir.glob("*.png"), key=lambda x: x.name)
        ordered_frame_names = [p.name for p in all_images[:num]]

    images = [input_dir / fname for fname in ordered_frame_names]
    valid_frames = [(cv2.imread(str(p)), n) for p, n in zip(images, ordered_frame_names) if cv2.imread(str(p)) is not None]
    
    if not valid_frames: 
        return
    
    h, w, _ = valid_frames[0][0].shape
    out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for frame, _ in valid_frames:
        out.write(frame)

    out.release()




def extract_exact_frames_from_video(video_path: Path, output_dir: Path, frame_map: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    for i in range(len(frame_map)):
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(output_dir / frame_map.get(str(i), f"{i}.png")), frame)
    cap.release()



def csv_to_h5(csv_path: str):
    csv = pd.read_csv(csv_path)

    if csv.empty or 'bodypart' not in csv.columns or 'frame_id' not in csv.columns:
        print(f"file is empty : {csv_path}")
        return pd.DataFrame()
    
    scorer = csv.get('labeller', ['default'])[0]
    bodyparts = csv["bodypart"].unique()
    images = sorted(csv['frame_id'].unique())

    data = {}
    for bp in bodyparts:
        for img in images:
            subset = csv[(csv['frame_id'] == img) & (csv['bodypart'] == bp)]
            x, y = (subset['x'].values[0], subset['y'].values[0]) if not subset.empty else (None, None)
            data.setdefault((scorer, bp, 'x'), []).append(x)
            data.setdefault((scorer, bp, 'y'), []).append(y)

    return pd.DataFrame(data, 
                        columns=pd.MultiIndex.from_tuples(data.keys(), 
                                                          names=['scorer', 'bodyparts', 'coords']),
                        index=pd.MultiIndex.from_tuples([("labeled-data", "output_video", i) for i in images]))
