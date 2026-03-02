import os, json, random
from bisect import bisect_right
from tqdm import tqdm

MAX_CHUNK = 30
DATA_SIZE_MUST_BE = 16968
IMG_ROOT = "/nas2/data/HPDv2/split"
org_path = "/nas2/data/HPDv2/train.json"

# -----------------------------
# Build per-chunk metadata + file membership sets
# -----------------------------
starts, ends, dir_paths, file_sets = [], [], [], []

for i in range(MAX_CHUNK):
    d = os.path.join(IMG_ROOT, f"out{i:03d}")
    if not os.path.isdir(d):
        raise ValueError(f"Directory does not exist: {d}")

    files = os.listdir(d)
    if not files:
        raise ValueError(f"No files found in: {d}")

    files.sort()  # zero-padded names -> lex sort works
    starts.append(int(os.path.splitext(files[0])[0]))
    ends.append(int(os.path.splitext(files[-1])[0]))
    dir_paths.append(d)
    file_sets.append(set(files))  # FAST existence check (handles holes)

VERY_LAST_IDX = max(ends)

def find_dir_if_exists(fname: str):
    """Return full path if file exists in its inferred chunk, else None."""
    try:
        idx = int(os.path.splitext(fname)[0])
    except ValueError:
        return None

    if idx > VERY_LAST_IDX:
        return None

    pos = bisect_right(starts, idx) - 1
    if pos < 0:
        return None

    # idx may fall into a gap between chunks
    if idx > ends[pos]:
        return None

    # handle non-contiguous filenames
    if fname not in file_sets[pos]:
        return None

    return os.path.join(dir_paths[pos], fname)

# -----------------------------
# Load original data
# -----------------------------
with open(org_path, "r") as f:
    org_data = json.load(f)

# -----------------------------
# Reservoir sampling (no huge memory, uniform random)
# -----------------------------
K = DATA_SIZE_MUST_BE
reservoir = []
seen = 0

append_res = reservoir.append
randint = random.randint

for item in tqdm(org_data):
    hp = item.get("human_preference")
    if not hp or len(hp) != 2:
        continue

    # exactly one chosen
    if hp[0] == hp[1] or (hp[0] + hp[1]) != 1:
        continue

    img0, img1 = item["image_path"][0], item["image_path"][1]
    p0 = find_dir_if_exists(img0)
    if p0 is None:
        continue
    p1 = find_dir_if_exists(img1)
    if p1 is None:
        continue

    chosen_path, reject_path = (p0, p1) if hp[0] == 1 else (p1, p0)

    ex = {
        "item_id": None,
        "t2i_category": "hpd_v2",
        "sub_category": "hpd_v2",
        "prompt": item["prompt"],
        "chosen": chosen_path,
        "rejected": reject_path,
    }

    seen += 1
    if len(reservoir) < K:
        append_res(ex)
    else:
        j = randint(1, seen)
        if j <= K:
            reservoir[j - 1] = ex

train_data = reservoir

print(f"Original data size: {len(org_data)}")
print(f"Eligible items seen: {seen}")
print(f"Training data size (final): {len(train_data)}")

# Assign item_id
for i, ex in enumerate(train_data):
    ex["item_id"] = f"{i:07d}"

# Save
save_path = f"/nas2/checkpoints/janus_dpo_rebuttal/data/external/newnew_train_hpd_v2_{len(train_data)}.json"
with open(save_path, "w") as f:
    json.dump(train_data, f, indent=4)
print(f"Training data saved to {save_path}")