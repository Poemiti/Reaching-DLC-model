
import yaml, os, json, cv2, csv, timeit
from typing import List, Dict, Literal, Optional, Any, ClassVar

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from PIL import Image
import cv2
import re





# ----------------------------------- for frame extraction -------------------------------------


def extract_frames_uniform(
            video_path: Path,
            output_dir: Path,
            num_frames: int,
            labeling_dir: str, ) -> list:
            
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Error opening video {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0 or total_frames == 0:
        cap.release()
        return []

    # Better uniform sampling
    # frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_indices = np.arange(0, total_frames, 63, dtype=int)

    # Extract rat name
    match = re.search(r"#\d+", str(video_path))
    rat_name = match.group(0) if match else "rat"
    rat_name = rat_name.replace("#", "")

    json_data = []
    label_studio_base = "http://10.24.12.180:8083/ls/"

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break

        base_name = f"frame{rat_name}_img{i+1}"
        frame_filename = output_dir / f"{base_name}.png"

        # Ensure unique filename
        counter = 1
        while frame_filename.exists():
            frame_filename = output_dir / f"{base_name}_{counter}.png"
            counter += 1

        cv2.imwrite(str(frame_filename), frame)

        rel_path = frame_filename.relative_to(output_dir.parent)

        json_data.append({
            "id": len(json_data) + 1,
            "data": {
                "frame_num": str(i + 1),
                "rel_img_path": str(rel_path),
                "label_studio_img_path": f"{label_studio_base}{labeling_dir}/Images/{frame_filename.name}",
                "source_video_filepath": str(video_path.resolve(),
                ),
            },
        })

    cap.release()
    return json_data


# phash method
def phash_distance(frame1, frame2):
    import imagehash

    pil1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    h1 = imagehash.phash(pil1)
    h2 = imagehash.phash(pil2)
    return h1 - h2


def extract_frames_phash(
            video_path: Path,
            output_dir: Path,
            max_frames: int,
            phash_threshold: int,
            labeling_dir: str,
            fps: int = 125,
            clip_duration_sec: int = 3,
            keep_duration_sec: int = 1, ) -> list :
            
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Problem opening video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    prev_frame = None
    json_data = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label_studio_base = "http://10.24.12.180:8083/ls"

    # Extract rat name safely
    match = re.search(r"#\d+", str(video_path))
    rat_name = match.group(0) if match else "rat"
    rat_name = rat_name.replace("#", "")

    # Clip logic
    clip_size = int(fps * clip_duration_sec)     # e.g. 375
    keep_frames = int(fps * keep_duration_sec)   # e.g. 125

    frame_idx = 0
    saved_count = 0

    while saved_count < max_frames and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Position inside current clip
        frame_in_clip = frame_idx % clip_size

        # Only keep frames from 0–1 sec of each clip
        if frame_in_clip < keep_frames:

            # pHash filtering
            if prev_frame is None or phash_distance(prev_frame, frame) > phash_threshold:

                base_name = f"frame{rat_name}_img{saved_count + 1}"
                frame_filename = output_dir / f"{base_name}.png"

                # Ensure unique filename
                counter = 1
                while frame_filename.exists():
                    frame_filename = output_dir / f"{base_name}_{counter}.png"
                    counter += 1

                cv2.imwrite(str(frame_filename), frame)

                rel_path = frame_filename.relative_to(output_dir.parent)

                json_data.append({
                    "id": len(json_data) + 1,
                    "data": {
                        "frame_num": str(saved_count + 1),
                        "rel_img_path": str(rel_path),
                        "label_studio_img_path": f"{label_studio_base}/{labeling_dir.stem}/Images/{frame_filename.name}",
                        "source_video_filepath": str(video_path.resolve()),
                    },
                })

                prev_frame = frame
                saved_count += 1

        frame_idx += 1

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




def copy_frames_to_dir(input_dir: Path, output_dir: Path) -> None: 
    import shutil
    
    for frame_path in input_dir.glob("*.png"):
        dest_path = output_dir / frame_path.name
        shutil.copy(frame_path, dest_path)



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
