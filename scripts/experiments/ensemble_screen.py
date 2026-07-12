# -*- coding: utf-8 -*-
"""DINOv2+DINOv3 skor ensemble taraması.
Mevcut dinov3 dev projeksiyonu (projection.pt) + dinov2 yedek projeksiyonu
(data/backups/2026-07-12_oncesi_ikizmaske) ile: skor = w*sim3 + (1-w)*sim2.
w, VAL kümesinde taranır; en iyi w TEST'te TEK KEZ ölçülür. Üretime dokunmaz."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"c:\Users\canya\Desktop\kapı")
HERE = ROOT / "data" / "train_cache" / "dinov3"
sys.path.insert(0, str(ROOT))
from search import load_config, load_excluded_trivial  # noqa: E402

K_VALUES = (1, 5, 10, 20)
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = load_config()

with open(ROOT / "data" / "eval" / "labels_clean.json", "r", encoding="utf-8-sig") as f:
    eslesme = json.load(f)["eslesme"]
with open(ROOT / "data" / "eval" / "projection_split.json", "r", encoding="utf-8-sig") as f:
    split = json.load(f)
excl = load_excluded_trivial(cfg) if cfg.get("exclude_trivial") else set()


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(768, 768), torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(768, 512))


def project(ckpt_path, mat):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    m = make_model().to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(mat), 4096):
            y = m(torch.from_numpy(mat[i:i + 4096]).to(device))
            outs.append(F.normalize(y, dim=1).cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["embeddings"].astype(np.float32), d["filenames"].tolist()


# --- dinov3 tarafı (güncel dev modeli)
c3, n3 = load_npz(HERE / "cad_clean_dinov3.npz")
p3, pn3 = load_npz(HERE / "photos_dinov3.npz")
# --- dinov2 tarafı (yedek dev modeli, ikiz-maskesiz round-3)
c2 = np.load(ROOT / "index" / "cad_embeddings.npy").astype(np.float32)
with open(ROOT / "index" / "cad_filenames.json", "r", encoding="utf-8") as f:
    n2 = json.load(f)
p2, pn2 = load_npz(HERE / "photos_dinov2.npz")

pos2 = {n: i for i, n in enumerate(n2)}
common = [n for n in n3 if n in pos2 and n not in excl]
keep3 = [i for i, n in enumerate(n3) if n in pos2 and n not in excl]
cad3 = project(ROOT / "data" / "models" / "projection.pt", c3[keep3])
cad2 = project(ROOT / "data" / "backups" / "2026-07-12_oncesi_ikizmaske" / "projection.pt",
               c2[[pos2[n] for n in common]])

ph3 = dict(zip(pn3, p3))
ph2 = dict(zip(pn2, p2))
name_pos = {n: i for i, n in enumerate(common)}


def eval_w(photos, w):
    usable = [(p, {Path(c).name for c in eslesme[p]
                   if Path(c).name in name_pos})
              for p in photos if p in eslesme and p in ph3 and p in ph2]
    usable = [(p, t) for p, t in usable if t]
    hits = {k: 0 for k in K_VALUES}
    q3 = project_photo_cache3 = np.stack([ph3[p] for p, _ in usable])
    q2 = np.stack([ph2[p] for p, _ in usable])
    q3 = proj_cache["q3"] if "q3" in proj_cache else proj_cache.setdefault(
        "q3", project(ROOT / "data" / "models" / "projection.pt", q3))
    q2 = proj_cache["q2"] if "q2" in proj_cache else proj_cache.setdefault(
        "q2", project(ROOT / "data" / "backups" / "2026-07-12_oncesi_ikizmaske" / "projection.pt", q2))
    for (p, targets), v3, v2 in zip(usable, q3, q2):
        scores = w * (cad3 @ v3) + (1 - w) * (cad2 @ v2)
        top = np.argpartition(-scores, 20)[:20]
        top = top[np.argsort(-scores[top])]
        tid = {name_pos[t] for t in targets}
        rank = next((r for r, i in enumerate(top, 1) if int(i) in tid), None)
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
    n = len(usable)
    return {k: hits[k] / n for k in K_VALUES}, n


for tag, photos in (("VAL", split["val"]),):
    proj_cache = {}
    print(f"--- {tag} w taraması (w = dinov3 ağırlığı)")
    best_w, best_r = None, -1
    for w in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        r, n = eval_w(photos, w)
        print(f"w={w:.2f}  " + "  ".join(f"r@{k} {r[k]:.1%}" for k in K_VALUES))
        if r[10] > best_r:
            best_r, best_w = r[10], w

proj_cache = {}
print(f"\nen iyi w={best_w} -> TEST (tek atış):")
r, n = eval_w(split["test"], best_w)
print(f"TEST ({n} foto)  " + "  ".join(f"r@{k} {r[k]:.1%}" for k in K_VALUES))
proj_cache = {}
r1, n = eval_w(split["test"], 1.0)
print(f"saf dinov3 kıyas " + "  ".join(f"r@{k} {r1[k]:.1%}" for k in K_VALUES))
