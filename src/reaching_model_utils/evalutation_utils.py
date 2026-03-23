import os
import re
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("pastel")


def select_file(root_path, title, filetype):
    root = tk.Tk()
    root.withdraw()  

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=root_path,
        filetypes=filetype
    )
    
    return file_path




def extract_dlc_folder(path):
    path = os.path.normpath(path)
    parts = path.split(os.sep)
    
    # Search for pattern DLC-{experimenter}-{date}
    for part in parts:
        if re.match(r"-\d{4}-\d{2}-\d{2}", part):
            return part
    
    return "Unknow-model"


# --------------------------------- plotting function -------------------------------------

def plot_loss(output_dir, df, train_col, test_col, title) : 

    plt.plot(df["step"], df[train_col], label="train", marker=".")
    plt.plot(df["step"], df[test_col], label="test", marker=".")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(output_dir / f"{title}.png")
    plt.close()


def plot_metric_epoch(output_dir, df, metric_col, title) : 

    plt.plot(df["step"], df[metric_col], marker=".")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.savefig(output_dir / f"{title}.png")
    plt.close()


def plot_metrics(output_dir, df, title) :
    ax = sns.barplot(data=df, x="metric", y="value", hue="dataset")
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.title(title)
    plt.ylabel("Error (px)")
    plt.savefig(output_dir / f"{title}.png")
    plt.close()
