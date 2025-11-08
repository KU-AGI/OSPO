# make it into preference data style
import os, json
from tqdm import tqdm
import random

dpath = "/nas2/data/Janus_dataset/next_v2/ablation/focusdiff/train_141330.json"
with open(dpath, "r") as f:
    data = json.load(f)

train_dataset = random.sample(data, 16000)
print(f"Total Length: {len(train_dataset)}")

save_dir = "/nas2/data/Janus_dataset/next_v2/ablation/focusdiff"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"train_{len(train_dataset)}.json")

with open(save_path, "w") as f:
    json.dump(train_dataset, f, indent=4)