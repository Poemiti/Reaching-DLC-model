import shutil
import os
import csv
import numpy as np

def normalize_annotation(annot, single_file):
    """Return unified (task_data, results) structure."""
    if single_file:
        task_data = annot.get('task', {}).get('data', {})
        results = annot.get('result', [])
    else:
        task_data = annot.get('data', {})
        results = annot.get('annotations', [{}])[0].get('result', [])
    return task_data, results


def extract_bodyparts(all_annotations, single_file):
    bodyparts = set()
    for annot in all_annotations:
        _, results = normalize_annotation(annot, single_file)
        for result in results:
            if result.get('type') == 'keypointlabels':
                bp = result.get('value', {}).get('keypointlabels', [''])[0]
                bodyparts.add(bp)
    return bodyparts


def build_frame_map(all_annotations, images_dir, single_file):
    frame_map = {}

    for i, annot in enumerate(all_annotations):
        task_data, _ = normalize_annotation(annot, single_file)

        rel_img = task_data.get('rel_img_path')
        if not rel_img:
            continue

        frame_name = os.path.basename(rel_img)
        key = str(i) if single_file else annot["id"]
        frame_map[key] = frame_name

        # copy image if needed
        full_img = images_dir / rel_img
        target_img = images_dir / frame_name
        if not target_img.exists() and full_img.exists():
            shutil.copy(full_img, target_img)

    return frame_map


def write_dlc_csv(csv_path, all_annotations, cfg, original_bodyparts, single_file):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['labeller', 'video_reference', 'frame_id', 'bodypart', 'x', 'y', 'confidence'])

        for annot in all_annotations:
            task_data, results = normalize_annotation(annot, single_file)

            video_ref = os.path.basename(os.path.dirname(task_data.get('source_video_filepath', '')))
            frame_id = os.path.basename(task_data.get('rel_img_path', f"{task_data.get('frame_num', 0)}.png"))

            for result in results:
                if result.get('type') != 'keypointlabels':
                    continue

                value = result.get('value', {})
                bp = value.get('keypointlabels', [''])[0]

                if bp not in original_bodyparts:
                    continue

                x = value.get('x', np.nan)
                y = value.get('y', np.nan)
                ow = result.get('original_width', 1)
                oh = result.get('original_height', 1)

                x, y = (x / 100 * ow), (y / 100 * oh)
                conf = 1 if not np.isnan(x) and not np.isnan(y) else np.nan

                writer.writerow([cfg.experimenter, video_ref, frame_id, bp, x, y, conf])