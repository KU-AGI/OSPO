import pandas as pd
import os
import io
import json
import hashlib
from tqdm import tqdm

def _guess_image_ext(b: bytes) -> str:
    if not b or len(b) < 12:
        return "bin"
    if b[:2] == b"\xff\xd8": return "jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n": return "png"
    if b[:6] in (b"GIF87a", b"GIF89a"): return "gif"
    if b[:4] == b"\x00\x00\x01\x00": return "ico"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP": return "webp"
    return "bin"

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _json_safe(obj):
    """Convert non-serializable types like Timestamps to strings."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return None
    return obj

def parquet_to_json_with_image_files(parquet_path, out_json_path=None, images_dir=None,
                                     image_cols: list = None, overwrite=False, use_jsonl=True):
    base_no_ext = os.path.splitext(parquet_path)[0]
    if out_json_path is None:
        out_json_path = base_no_ext + (".jsonl" if use_jsonl else ".json")
    if images_dir is None:
        images_dir = base_no_ext + "_images"
    os.makedirs(images_dir, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    length = len(df)
    print(f"# Total rows: {length}")

    # auto-detect image/bytes columns
    if image_cols is None:
        image_cols = [
            c for c in df.columns
            if df[c].apply(lambda x: isinstance(x, (bytes, bytearray))).any()
        ]
    print(f"# Image columns: {image_cols if image_cols else 'None'}")

    saved_files = 0
    if use_jsonl:
        out_f = open(out_json_path, "w", encoding="utf-8")
    else:
        records_buffer = []

    for _, row in df.iterrows():
        rec = row.to_dict()

        for col in image_cols:
            val = rec.get(col)
            if isinstance(val, (bytes, bytearray)) and len(val) > 0:
                b = bytes(val)
                h = _hash_bytes(b)
                ext = _guess_image_ext(b)
                fpath = os.path.join(images_dir, f"{h}.{ext}")
                if not (os.path.exists(fpath) and not overwrite):
                    with open(fpath, "wb") as f:
                        f.write(b)
                        saved_files += 1
                # rec[col] = os.path.relpath(fpath, start=os.path.dirname(out_json_path) or ".")
                rec[col] = os.path.abspath(fpath)

        # make everything JSON-safe
        rec = {k: _json_safe(v) for k, v in rec.items()}

        if use_jsonl:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            records_buffer.append(rec)

    if use_jsonl:
        out_f.close()
    else:
        # ensure timestamps are handled
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(records_buffer, f, ensure_ascii=False, indent=2, default=_json_safe)

    print(f"# Saved {saved_files} image file(s) into: {images_dir}")
    print(f"# Wrote {'JSONL' if use_jsonl else 'JSON'} to: {out_json_path}")

    return length


# --- Example usage ---
if __name__ == "__main__":

    pq_dir = "/nas2/data/pickapic_v2"
    img_path="/nas2/data/pickapic_v2/images"
    os.makedirs(img_path, exist_ok=True)
        
    pq_path_list = [f for f in os.listdir(pq_dir) if f.startswith("train") and f.endswith(".parquet")]
    # pq_path="/nas2/data/pickapic_v2/train-00000-of-00645-b66ac786bf6fb553.parquet"

    count = 0

    for idx, pq_path in tqdm(enumerate(pq_path_list)):

        load_path = os.path.join(pq_dir, pq_path)
        json_path=f"/nas2/data/pickapic_v2/data/train_{idx:05d}.json"
        # os.makedirs(os.path.dirname(json_path), exist_ok=True)

        pq_len = parquet_to_json_with_image_files(
            parquet_path=load_path,
            out_json_path=json_path,         # defaults to data/input_file.jsonl
            images_dir=img_path,             # defaults to data/input_file_images/
            image_cols=["jpg_0", "jpg_1"],   # auto-detect bytes columns
            overwrite=False,
            use_jsonl=False,            # one JSON object per line
        )
        count += pq_len

    print(f"Total Length: {count}")