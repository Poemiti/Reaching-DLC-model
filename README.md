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

### 0. Setup the python environment

1. Install conda : https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install  
2. Create the environment using this line in your terminal :  
```bash
conda env create --file environment.yml
```  

Once everything is set up, every time you will want to use this code, activate the `DEEPLABCUT` environment using this line:   
```bash
conda activate DEEPLABCUT
```  

### 1. Modify the `config.yaml` file  

Modify every parameters if necessary.   
All parameters must follow the Pydantic BaseModel setup in
`src/reaching_model_utils/config.py`.   
New parameters can be added as well, don`t forget to add them to the BaseModel

### 2. Run each `src/` files in there order

Like this : 
```bash
python3 src/sript.py
```

#### `0.initialisation`

It creates the folder system if it doesn`t already exist

#### `1.1.extract_frames.py`

Extract frames from the videos absolute path listed in `video_to_extract` from the config.yaml.  
The script will extract the number of frames set in `num_frames_per_video` following the `extract_method` set. 2 Methods are available : **phash** and **uniform**.  

After extracting the frame, those frame must be labeled using **Label Studio**.   
After labelling, export the data using the `export` button on the top left. Then select the first option given. The file exported must be stored in the labeling path set in the config file, and the name must be `frames_annotations_meta.json`. (e.g. */media/filer2/T4b/Labeling/Model_Poe/frames_annotations_meta.json*)

*(The name can be changed, but you will have to change the code as well)*


#### `1.2.annotation_verification.py`

This script verify if the labeling was correct. It will tell which frame is missing labelling, or if their was bodypart duplication.  
If so, go back to label studio to fixe it. If everything is correct, pass to the next script.

#### `2.training.py`

For the training the only parameters that need to be checked:  
- batch_size : If your computer doesn't have a good GPU, lower this number  
- num_frames_for_train : Remember, to train a model, only train on 90% or 95% of your whole dataset. The rest of your frames will be used for evaluating the performance.
- `info_skeleton.yaml` : the bodypart written in there will be the ones the model will train on. If your frame_annotations_meta.json contains more bodypart labelled, it alright, the model will only look at the bodypart set in the info_skeleton.  

The training can take hours, even days ! A very useful line code to use (in the terminal) is this :
```bash
nohup python3 -u src/2.training.py > main.out &
tail -f main.out 
```
This line will make the code work all day and night even if you are log out of your session. If the computer is shut down, it will stop tho.

#### `3.evaluation.py`

This script will produce figure to tell if the model trained correctly. Indeed the following figures will be produce (in *data/evaluation/YourModel/*):   
- Loss curve  
- Recall  
- RMSE (Overall error rate)  
- Bodypart Error (%) or (px) 

#### `3.refine_model.py`

Not implemented yet...

