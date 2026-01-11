import pandas as pd
import os, io, json, hashlib, shutil
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _guess_image_ext_from_magic(b: bytes) -> str:
    if not b or len(b) < 12:
        return "bin"
    if b[:2] == b"\xff\xd8": return "jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n": return "png"
    if b[:6] in (b"GIF87a", b"GIF89a"): return "gif"
    if b[:4] == b"\x00\x00\x01\x00": return "ico"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP": return "webp"
    return "bin"

def _json_safe(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # PIL.Image cannot be serialized; we should have converted it to a path already.
    return obj

def _image_to_bytes(img: Image.Image) -> bytes:
    """Serialize PIL Image to bytes (prefer its format, default PNG)."""
    bio = io.BytesIO()
    fmt = (img.format or "PNG")
    img.save(bio, format=fmt)
    return bio.getvalue()

def _save_single_value_as_image(val, images_dir, overwrite=False):
    """
    Accepts: bytes/bytearray, PIL.Image.Image, str (existing local path), dict with 'bytes',
             returns absolute path string. For unknown types returns None (leave as-is).
    """
    # case 1: raw bytes
    if isinstance(val, (bytes, bytearray, memoryview)):
        b = bytes(val)
        h = _hash_bytes(b)
        ext = _guess_image_ext_from_magic(b)
        fpath = os.path.abspath(os.path.join(images_dir, f"{h}.{ext}"))
        if overwrite or not os.path.exists(fpath):
            with open(fpath, "wb") as f:
                f.write(b)
        return fpath

    # case 2: PIL Image
    if isinstance(val, Image.Image):
        b = _image_to_bytes(val)
        h = _hash_bytes(b)
        # try to honor original format
        ext = (val.format.lower() if val.format else _guess_image_ext_from_magic(b))
        if ext == "jpeg": ext = "jpg"
        fpath = os.path.abspath(os.path.join(images_dir, f"{h}.{ext}"))
        if overwrite or not os.path.exists(fpath):
            with open(fpath, "wb") as f:
                f.write(b)
        return fpath

    # case 3: local string path → copy (to keep everything under images_dir)
    if isinstance(val, str) and os.path.exists(val):
        # read bytes, hash, re-save
        try:
            with open(val, "rb") as rf:
                b = rf.read()
            h = _hash_bytes(b)
            ext = os.path.splitext(val)[1].lstrip(".").lower() or _guess_image_ext_from_magic(b)
            if ext == "jpeg": ext = "jpg"
            fpath = os.path.abspath(os.path.join(images_dir, f"{h}.{ext}"))
            if overwrite or not os.path.exists(fpath):
                with open(fpath, "wb") as wf:
                    wf.write(b)
            return fpath
        except Exception:
            # fall back to just returning the original string
            return os.path.abspath(val)

    # case 4: dict-like with raw bytes
    if isinstance(val, dict):
        # common patterns: {'bytes': ...} or {'array': ...}, we support only bytes gracefully
        b = val.get("bytes", None)
        if isinstance(b, (bytes, bytearray, memoryview)):
            return _save_single_value_as_image(b, images_dir, overwrite=overwrite)
        # if dict has 'path', treat like string path
        p = val.get("path", None)
        if isinstance(p, str):
            return _save_single_value_as_image(p, images_dir, overwrite=overwrite)
        return None

    # everything else (e.g., URLs) → return as-is (no downloading here)
    return None

def _save_value_maybe_list(val, images_dir, overwrite=False):
    """Save a single image or list of images; return path or list of paths, or None."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        out = []
        for v in val:
            p = _save_single_value_as_image(v, images_dir, overwrite=overwrite)
            out.append(p if p is not None else v)  # preserve original if unsupported
        return out
    else:
        p = _save_single_value_as_image(val, images_dir, overwrite=overwrite)
        return p if p is not None else val  # preserve original if unsupported

def parquet_to_json_with_image_files(
    parquet_path,
    out_json_path=None,
    images_dir=None,
    image_cols: list = None,
    overwrite=False,
    use_jsonl=True,
    autodetect_images=True,
    autodetect_sample_rows=128,
):
    base_no_ext = os.path.splitext(parquet_path)[0]
    if out_json_path is None:
        out_json_path = base_no_ext + (".jsonl" if use_jsonl else ".json")
    if images_dir is None:
        images_dir = base_no_ext + "_images"
    os.makedirs(images_dir, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    length = len(df)
    print(f"# Total rows: {length}")

    # Auto-detect image-ish columns if requested
    if image_cols is None and autodetect_images:
        sample = df.head(min(autodetect_sample_rows, length))
        def looks_like_image_col(series: pd.Series) -> bool:
            # bytes or PIL.Image in any of the sample values, or lists of those
            for v in series:
                if isinstance(v, (bytes, bytearray, memoryview, Image.Image)):
                    return True
                if isinstance(v, (list, tuple)) and any(isinstance(x, (bytes, bytearray, memoryview, Image.Image)) for x in v):
                    return True
                if isinstance(v, str) and os.path.exists(v):
                    return True
                if isinstance(v, dict) and ("bytes" in v or "path" in v):
                    return True
            return False
        image_cols = [c for c in df.columns if looks_like_image_col(sample[c])]
    print(f"# Image columns: {image_cols if image_cols else 'None'}")

    saved_files = 0
    # we’ll count by comparing existing file count before/after if you need exact numbers,
    # but here we simply report how many times we wrote new files if overwrite=True.
    wrote_paths = set()

    if use_jsonl:
        out_f = open(out_json_path, "w", encoding="utf-8")
    else:
        records_buffer = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        rec = row.to_dict()

        if image_cols:
            for col in image_cols:
                if col in rec:
                    new_val = _save_value_maybe_list(rec[col], images_dir, overwrite=overwrite)
                    # Count newly created paths in this run (best-effort)
                    if isinstance(new_val, str) and new_val and new_val.startswith(os.path.abspath(images_dir)):
                        if new_val not in wrote_paths:
                            wrote_paths.add(new_val)
                            saved_files += 1
                    elif isinstance(new_val, (list, tuple)):
                        for p in new_val:
                            if isinstance(p, str) and p and p.startswith(os.path.abspath(images_dir)):
                                if p not in wrote_paths:
                                    wrote_paths.add(p)
                                    saved_files += 1
                    rec[col] = new_val

        # make JSON-safe (timestamps etc.)
        rec = {k: _json_safe(v) for k, v in rec.items()}

        if use_jsonl:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            records_buffer.append(rec)

    if use_jsonl:
        out_f.close()
    else:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(records_buffer, f, ensure_ascii=False, indent=2, default=_json_safe)

    print(f"# Saved ~{saved_files} image file(s) into: {os.path.abspath(images_dir)}")
    print(f"# Wrote {'JSONL' if use_jsonl else 'JSON'} to: {os.path.abspath(out_json_path)}")
    return length

# --- Example usage for AnyEdit ---
if __name__ == "__main__":
    pq_dir = "/nas2/data/AnyEdit/split"
    img_path = "/nas2/data/AnyEdit/split/images"
    os.makedirs(img_path, exist_ok=True)

    # pq_path_list = [f for f in os.listdir(pq_dir) if f.startswith("train") and f.endswith(".parquet")]
    pq_path_list = [f for f in os.listdir(pq_dir) if f.endswith(".parquet")]

    count = 0
    for idx, pq_path in tqdm(list(enumerate(pq_path_list)), desc="Files"):
        load_path = os.path.join(pq_dir, pq_path)
        json_path = f"{pq_dir}/train_{idx:05d}.json"

        pq_len = parquet_to_json_with_image_files(
            parquet_path=load_path,
            out_json_path=json_path,
            images_dir=img_path,
            image_cols=["image_file", "edited_file"], # , "visual_input"],  # <- include visual_input
            overwrite=False,
            use_jsonl=False,   # single JSON array file; set True for .jsonl
            autodetect_images=False,  # we explicitly pass image_cols
        )
        count += pq_len

    print(f"Total Length: {count}")