
from pathlib import Path
import pandas as pd
import reaching_model_utils.evaluation_utils as evaluation_utils
from reaching_model_utils.config import load_config
import sys


'''

mAP = mean average precision = how correct prediction are
mAR = mean average recall = how many true point the model found

'''

# ----------------------------- re run the DLC evaluation if necessary  -----------------------------

root = "/media/filer2/T4b/UserFolders/Poemiti"

launch_dlc_eval = input("Do you need to evaluate a model (deeplabcut.evaluate_network) ? (y/n) : ")

if launch_dlc_eval == "y" : 
    print("Select the config file correponding to the model to evaluate")
    
    config_path = evaluation_utils.select_file(root, title="Select Config.yaml of the model to evalutate",
                                          filetype=[("YAML files", "*.yaml")])
    
    if config_path : 
        import deeplabcut
        from deeplabcut.pose_estimation_pytorch import set_load_weights_only
        set_load_weights_only(False)

        deeplabcut.evaluate_network(config_path, Shuffles=[1],
                                plotting=True,
                                per_keypoint_evaluation=True)
    else : 
        print("No file selected, stop!")
        sys.exit()


# ----------------------------- select files  -----------------------------


print("Select Learning stat csv file (dlc-models-pytorch/.../learning_stat.csv)")
model_learning_stat_path = evaluation_utils.select_file(root, title="Select Learning stat (dlc-models-pytorch/.../learning_stat.csv)",
                                             filetype=[("CSV files", "*.csv*")])

print("Select Evaluation metrics csv file (evaluation-results-pytorch/.../CombinedEvaluation-results.csv)")
model_evaluation_path = evaluation_utils.select_file(root, title="Select Evaluation metrics (evaluation-results-pytorch/.../CombinedEvaluation-results.csv)",
                                             filetype=[("CSV files", "*.csv*")])

print("Select Evaluation metrics per bodyparts (evaluation-results-pytorch/.../...keypoint-results.csv)")
perbody_evaluation_path = evaluation_utils.select_file(root, title="Select Evaluation metrics per bodyparts (evaluation-results-pytorch/.../...keypoint-results.csv)",
                                             filetype=[("CSV files", "*.csv*")])



# model_learning_stat_path = "/media/filer2/T4b/UserFolders/Poemiti/Reaching-DLC-model-main/data/model/DLC-Poe-2026-03-26/dlc-models-pytorch/iteration-0/DLCMar26-trainset95shuffle1/train/learning_stats.csv"
# model_evaluation_path = "" # "/media/filer2/T4b/UserFolders/Poemiti/Reaching-DLC-model-main/data/model/DLC-Poe-2026-03-26/evaluation-results-pytorch/iteration-0/DLCMar26-trainset95shuffle1/DLC_Resnet50_DLCMar26shuffle1_snapshot_260-results.csv"
# perbody_evaluation_path = "/media/filer2/T4b/UserFolders/Poemiti/Reaching-DLC-model-main/data/model/DLC-Poe-2026-03-26/evaluation-results-pytorch/iteration-0/DLCMar26-trainset95shuffle1/DLC_Resnet50_DLCMar26shuffle1_snapshot_260-keypoint-results.csv"

# create folder in which the evaluation plot will be saved

if model_learning_stat_path or model_evaluation_path or perbody_evaluation_path: 
    cfg = load_config("../config.yaml")
    model_folder_name = evaluation_utils.extract_dlc_folder(model_learning_stat_path)
    output_dir = cfg.paths.evaluation / model_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation will be saved in {output_dir}")
else : 
    print("No file selected, stop!")
    sys.exit()


# ----------------------------- plot and save -----------------------------

if model_learning_stat_path:

    df_loss = pd.read_csv(model_learning_stat_path)

    evaluation_utils.plot_loss(output_dir, df_loss, "losses/train.total_loss", "losses/eval.total_loss", "Loss")
    evaluation_utils.plot_metric_epoch(output_dir, df_loss, "metrics/test.rmse", "test_RMSE_over_epochs")



if model_evaluation_path: 
    
    df_eval = pd.read_csv(model_evaluation_path)

    rmse_cols = ["train rmse", "test rmse", "train rmse_pcutoff", "test rmse_pcutoff"]
    recall_cols = ["train mAP", "test mAP", "train mAR", "test mAR"]

    rmse = []
    recall = []

    for c in rmse_cols : 
        dataset, metric = c.split(" ")
        rmse.append({
            "dataset" : dataset,
            "metric" : metric,
            "value" : df_eval[c].values[0]
        })

    for c in recall_cols : 
        dataset, metric = c.split(" ")
        recall.append({
            "dataset" : dataset,
            "metric" : metric,
            "value" : df_eval[c].values[0]
        })

    df_rmse = pd.DataFrame(rmse)
    df_recall = pd.DataFrame(recall)

    evaluation_utils.plot_metrics(output_dir, df_rmse, "RMSE")
    evaluation_utils.plot_metrics(output_dir, df_recall, "Recall")


if perbody_evaluation_path: 

    def build_bodypart_error(df, train, testt):
        df.extend([
            {
                "bodypart": bp,
                "type": "train",
                "value": round(train, 1),
            },
            {
                "bodypart": bp,
                "type": "test",
                "value": round(test, 1),
            }
        ])

    df_body = pd.read_csv(perbody_evaluation_path)

    bodypart_error = []
    bodypart_error_percentage = []

    for bp in df_body.columns[1:] : 
        train = df_body[bp][0]
        test = df_body[bp][1]

        build_bodypart_error(bodypart_error, train, test)
        build_bodypart_error(bodypart_error_percentage, (train*100)/512, (test*100)/512)

    
    
    evaluation_utils.plot_bodypart_error(output_dir, 
                                         pd.DataFrame(bodypart_error), 
                                         "Bodypart Error (px)")
    evaluation_utils.plot_bodypart_error(output_dir, 
                                         pd.DataFrame(bodypart_error_percentage), 
                                         "Bodypart Error (%)")
    


