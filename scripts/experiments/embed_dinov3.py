# -*- coding: utf-8 -*-
"""DINOv3 vitb16 tarama deneyi — adım 1: embedding üretimi.
cad_clean + cad_hedthick5 (32k'şar) ve etiketli foto lineart'ları (eval cache)
DINOv3 ile embedlenir; çıktılar bu klasöre npz olarak yazılır (checkpoint'li).
Üretime DOKUNMAZ; her şey data/train_cache/dinov3 altında kalır."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "train_cache" / "dinov3"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMG_SIZE = 512          # patch16 katı; dinov2'deki 518'e en yakın adil çözünürlük
BATCH = 12

from transformers import AutoImageProcessor, AutoModel  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(MODEL_ID)
proc.size = {"height": IMG_SIZE, "width": IMG_SIZE}
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval().to(device)
print(f"model yüklendi: {MODEL_ID} | {device} | {IMG_SIZE}px")


def embed_batch(pil_images):
    with torch.no_grad():
        inputs = proc(images=pil_images, return_tensors="pt").to(device)
        inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        out = model(**inputs)
        emb = out.pooler_output.float().cpu().numpy().astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb


def embed_list(paths, names, out_path: Path):
    ckpt = out_path.with_suffix(".part.npz")
    chunks, start = [], 0
    if ckpt.exists():
        try:
            d = np.load(ckpt, allow_pickle=False)
            done = d["filenames"].tolist()
            if done == names[:len(done)]:
                chunks, start = [d["embeddings"]], len(done)
                print(f"  checkpoint: {start} hazır, devam")
        except Exception:
            pass
    pbar = tqdm(total=len(names), initial=start, desc=out_path.stem, unit="img")
    since = 0
    for i in range(start, len(names), BATCH):
        batch = []
        for p in paths[i:i + BATCH]:
            with Image.open(p) as im:
                im.load()
                batch.append(im.convert("RGB"))
        chunks.append(embed_batch(batch))
        pbar.update(len(batch))
        since += 1
        if since >= 50:
            emb = np.concatenate(chunks, axis=0)
            chunks = [emb]
            np.savez(ckpt, embeddings=emb, filenames=np.array(names[:len(emb)]))
            since = 0
    pbar.close()
    emb = np.concatenate(chunks, axis=0).astype(np.float32)
    np.savez(out_path, embeddings=emb, filenames=np.array(names))
    if ckpt.exists():
        ckpt.unlink()
    print(f"  -> {out_path.name} {emb.shape}")


# ortak CAD listesi: base ve hedthick5 indekslerinin kesişimi (07 ile aynı mantık)
name_lists = []
for d in (ROOT / "index", ROOT / "index" / "variants" / "hedthick5_vitb14"):
    with open(d / "cad_filenames.json", "r", encoding="utf-8") as f:
        name_lists.append(json.load(f))
common = sorted(set(name_lists[0]) & set(name_lists[1]))
print(f"{len(common)} ortak CAD")

# 1) cad_clean
clean_dir = ROOT / "data" / "cad_clean"
if not (OUT / "cad_clean_dinov3.npz").exists():
    embed_list([clean_dir / n for n in common], common, OUT / "cad_clean_dinov3.npz")

# 2) cad_hedthick5
hed_dir = ROOT / "data" / "variants" / "cad_hedthick5"
if not (OUT / "cad_hed_dinov3.npz").exists():
    embed_list([hed_dir / n for n in common], common, OUT / "cad_hed_dinov3.npz")

# 3) etiketli foto lineart'ları (eval cache'ten)
with open(ROOT / "data" / "eval" / "labels_clean.json", "r", encoding="utf-8-sig") as f:
    photos = sorted(json.load(f)["eslesme"])
cache_dir = ROOT / "data" / "eval" / "cache"
have = [p for p in photos if (cache_dir / (p + ".lineart.png")).exists()]
print(f"{len(have)}/{len(photos)} fotoğrafın lineart cache'i var")
if not (OUT / "photos_dinov3.npz").exists():
    embed_list([cache_dir / (p + ".lineart.png") for p in have], have,
               OUT / "photos_dinov3.npz")

print("EMBED TAMAM")
