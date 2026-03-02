import os, json

dpath1 = "/home/yjoh/project/OSPO/ospo/rebuttal/compute/stage2_abl_data1_train.json"
dpath2 = "/home/yjoh/project/OSPO/ospo/rebuttal/compute/stage2_abl_data2_train.json"

with open(dpath1, "r") as f1, open(dpath2, "r") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

data = data1 + data2

save_path = "/nas2/checkpoints/janus_dpo_rebuttal/data/stage2_abl"
os.makedirs(save_path, exist_ok=True)  
with open(os.path.join(save_path, f"stage2_abl_train_{len(data)}.json"), "w") as f:
    json.dump(data, f, indent=4)