
import yaml, os, json, cv2, csv, timeit
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

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
