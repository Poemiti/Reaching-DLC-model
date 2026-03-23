
from pathlib import Path
import pandas as pd
import reaching_model_utils.video_utils as video_utils
import sys


'''

mAP = mean average precision = how correct prediction are
mAR = mean average recall = how many true point the model found

'''



# ----------------------------- select files  -----------------------------

root = "/media/filer2/T4b/"
model_learning_stat_path = video_utils.select_file(root, title="Select Learning stat (dlc-models-pytorch/.../learning_stat.csv)",
                                             filetype=[("CSV files", "*.csv*")])
model_evaluation_path = video_utils.select_file(root, title="Select Evaluation metrics (evaluation-results-pytorch/.../CombinedEvaluation-results.csv)",
                                             filetype=[("CSV files", "*.csv*")])

# create folder in which the evaluation plot will be saved

if model_learning_stat_path or model_evaluation_path: 
    model_folder_name = video_utils.extract_dlc_folder(model_learning_stat_path)
    output_dir = Path(f"./{model_folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
else : 
    print("No file selected, stop!")
    sys.exit()


# ----------------------------- plot and save -----------------------------

if model_learning_stat_path:

    df_loss = pd.read_csv(model_learning_stat_path)

    for col in df_loss.columns : 
        print(col)

    video_utils.plot_loss(output_dir, df_loss, "losses/train.total_loss", "losses/eval.total_loss", "Loss")
    video_utils.plot_metric_epoch(output_dir, df_loss, "metrics/test.rmse", "test_RMSE_over_epochs")



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

    video_utils.plot_metrics(output_dir, df_rmse, "RMSE")
    video_utils.plot_metrics(output_dir, df_recall, "Recall")

