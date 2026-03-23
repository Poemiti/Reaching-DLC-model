from pathlib import Path
import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------- setup path and variable --------------------------------------

root = Path("./Labeling")
annot_dir = root / "Annotations"
image_dir = root / "Images"
model_dir = Path("/media/filer2/T4b/Models/DLC/REJANE_rat_right_model-2025-06-18/Modelconfig_predict_24_200_1000/DLC-project-2025-06-17/labeled-data/output_video/")


print(f"\n{annot_dir}: {annot_dir.exists()}")
print(f"{image_dir}: {image_dir.exists()}")

overall_data = []

metadata_path_list = list(annot_dir.glob("*"))
image_path_list = list(image_dir.glob("*.png"))
model_image_list = list(model_dir.glob("*.png"))

print(f"\nNumber of annotation: {len(metadata_path_list)} ")
print(f"Number of images: {len(image_path_list)}")


# -------------------------------------- count number of curretly used annotation --------------------------------------


model_data = {}
for file in model_image_list:
    name = re.sub(r"frame", "", file.stem.split("_")[0])

    if name not in model_data:
        model_data[name] = 1
    else:
        model_data[name] += 1

# -------------------------------------- count number of all annotation available --------------------------------------

for path in metadata_path_list : 
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rel_img_path = root / meta.get('task', {}).get('data', {}).get('rel_img_path')
    rat = re.sub(r"frame", "", rel_img_path.stem.split("_")[0])

    overall_data.append({
        "annotation" : path.stem,
        "image" : rel_img_path.stem,
        "image_exist?" : rel_img_path in image_path_list,
        "rat" : rat,
    })


# -------------------------------------- plots the counts for visualisation --------------------------------------

df = pd.DataFrame(overall_data)
df_model = pd.DataFrame(list(model_data.items()), columns=["rat", "count"])

# Filter data
df_filtered = df[df["image_exist?"] == True]


ax = sns.countplot(data=df_filtered, x="rat", color="lightblue", label="Available annotation")
ax.bar_label(ax.containers[0])

ax2 = sns.barplot(data=df_model, x="rat", y="count", color="salmon", label="Current annotation used")
ax2.bar_label(ax2.containers[1])

plt.legend(title="Data type")
plt.xlabel("Rat")
plt.ylabel("Count")
plt.title(f"Distribution of Annotated rats\nCurrent / Available : {len(model_image_list)} / {len(metadata_path_list)}")
plt.show()