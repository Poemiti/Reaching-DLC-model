import os
import re
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")

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
    
    # Search for pattern {project}-{experimenter}-{date}
    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9_]+-[A-Za-z0-9_]+-\d{4}-\d{2}-\d{2}", part):
            return part
    
    return "Unknow-model"


# --------------------------------- plotting function -------------------------------------

def plot_loss(output_dir, df, train_col, test_col, title) : 

    fig, ax = plt.subplots(figsize=(10, 9))
    # fig.subplots_adjust(left=0.3, bottom=0.2)

    ax.plot(df["step"], df[train_col], label="train", marker=".")
    ax.plot(df["step"], df[test_col], label="test", marker=".")

    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    fig.tight_layout()
    plt.savefig(output_dir / f"{title}.png")
    plt.savefig(output_dir / f"{title}.svg")

    plt.close()


def plot_metric_epoch(output_dir, df, metric_col, title) : 

    fig, ax = plt.subplots(figsize=(10, 9))
    # fig.subplots_adjust(left=0.2, bottom=0.2)

    ax.plot(df["step"], df[metric_col], marker=".")

    plt.xlabel("Epoch")
    plt.ylabel(title)

    fig.tight_layout()
    plt.savefig(output_dir / f"{title}.png")
    plt.savefig(output_dir / f"{title}.svg")

    plt.close()


def plot_metrics(output_dir, df, title) :

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=0.15, bottom=0.2)

    sns.barplot(data=df, x="metric", y="value", hue="dataset", ax=ax)
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])

    plt.legend(loc="lower right")
    plt.title(title)
    plt.ylabel("Error (px)")

    fig.tight_layout()
    plt.savefig(output_dir / f"{title}.png")
    plt.savefig(output_dir / f"{title}.svg")

    plt.close()


def plot_bodypart_error(output_dir, df, title) : 

    fig, ax = plt.subplots(figsize=(10, 12))
    ax = sns.barplot(data=df, x="value", y="bodypart", hue="type")
    ax.bar_label(ax.containers[0], padding=2, fontsize=11)
    ax.bar_label(ax.containers[1], padding=2, fontsize=11)
    fig.subplots_adjust(left=0.3)
    plt.title(title)
    plt.ylabel("Error")

    fig.tight_layout()
    plt.savefig(output_dir / f"{title}.png")
    plt.savefig(output_dir / f"{title}.svg")

    plt.close()
