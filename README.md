## DLC model training on reaching task videos

This repository present a pipeline to :  
1. extract frames for a list of videos
2. label frames using [Label-studio](https://labelstud.io/)
3. train a [DeepLabCut](https://github.com/DeepLabCut) model on those labeled frames 
4. evaluate any deeplabcut project permformance

This pipelines doesn't use the labeling system provided by DeepLabCut (DLC).  
Instead it uses Label-Studio, an online labeling system easy to use.  
Since the labeling system is different, this pipeline adapts the outputs of  
label studio, to be compatible with DLC.

## Pipeline structure

```bash

| config.yaml                       # project configuration, where the paths are
| info_skeleton.yaml                # info about labeling and the skeleton

| -- src/
    | -- reaching_model_utils/      # where all the functions are
        | config.py
        | evaluation.utils.py
        | video_utils.py
    
    | 0.initialisation.py           # setup the paths (put in the config.yaml)
    | 1.1.extract_frames.py
    | 1.2.annotation_verification.py
    | 2.training.py
    | 3.evaluation.py
    | count.py                      # small script to count proportion of each rats in the extracted frames

| -- data/                          # where all the outputs falls

    | -- labelling/                 # where the extracted frames will be (can be somewhere else)
        | -- Annotations/
        | -- Images/
        | frame_metadata.json
    
    | -- model/                     # where all the trained models will be
        | annotation_list.json  
        | -- DLC-project-03-26-26/  # the actual deeplabcut project folder (created by deeplabcut.create_project)
            | ...

    | -- evaluation /               # where the evaluation figures will be
        | -- DLC-project-03-26-26/
            | Loss.png
            | Recall.png
            | RMSE.png
        | -- ...    

    | -- temporary/                 # temporary folder for the video created for DLC
        | output_video.mp4
```


## How to use this pipeline ? 

### 1. Modify the `config.yaml` file  

Modify every parameters if necessary.   
All parameters must follow the Pydantic BaseModel setup in
`src/reaching_model_utils/config.py`.   
New parameters can be added as well, don't forget to add them to the BaseModel

### 2. Run each `src/` files in there order

#### `0.initialisation`

It creates the folder system if it doesn't already exist

#### `1.1.extract_frames.py`

