import os, json
from tqdm import tqdm

ddir = "/nas2/data/pickapic_v2/data"

# data merge
merged = []
for f in os.listdir(ddir):
    load_path = os.path.join(ddir, f)
    with open(load_path, "r") as f:
        data = json.load(f)
    merged.extend(data)


# data construct as preference data
train_dataset = []
for d in merged:

    # best_image_uid 의 이미지가 chosen
    best_image_uid = d['best_image_uid']

    if best_image_uid == d['image_0_uid']:
        cpath = d['jpg_0']
        rpath = d['jpg_1']
    elif best_image_uid == d['image_1_uid']:
        cpath = d['jpg_1']
        rpath = d['jpg_0']
    else:
        print("Error")
        continue

    if len(train_dataset) < 17000:
        item_id = d['best_image_uid'].split("-")[0]
        train_dataset.append({
                "item_id": item_id,
                "t2i_category": "Pickapic2",
                "sub_category": "Pickapic2",
                "prompt": d['caption'],
                "chosen": cpath,
                "rejected": rpath
            })
    else:
        break

print(f"Total Length: {len(train_dataset)}")


# save merge
merge_save_dir = ddir
train_save_dir = "/nas2/data/Janus_dataset/next_v2/ablation/pickapic2"
os.makedirs(merge_save_dir, exist_ok=True)
os.makedirs(train_save_dir, exist_ok=True)

save_path = os.path.join(merge_save_dir, f"merged_00_12_{len(merged)}.json")
with open(save_path, "w") as f:
    json.dump(merged, f, indent=2)


save_path = os.path.join(train_save_dir, f"train_{len(train_dataset)}.json")
with open(save_path, "w") as f:
    json.dump(train_dataset, f, indent=4)

