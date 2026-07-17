# -*- coding: utf-8 -*-
"""Ensemble tur 2: güncel dinov2 dev (projection.pt) + güncel dinov3 dev
(projection_dev_dinov3_r4.pt). w VAL'de taranır, en iyi w TEST'te tek kez
ölçülür; ek olarak gruplu kart isabeti (ham dinov2 ikiz tekilleştirme, 0.95)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "data" / "train_cache" / "dinov3"
sys.path.insert(0, str(ROOT))
from search import load_config, load_excluded_trivial  # noqa: E402

K_VALUES = (1, 5, 10, 20)
DUP_SIM = 0.95
CKPT3 = ROOT / "data" / "models" / "projection_dev_dinov3_r4.pt"
CKPT2 = ROOT / "data" / "models" / "projection.pt"   # az önce eğitilen dinov2 dev
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


c3, n3 = load_npz(HERE / "cad_clean_dinov3.npz")
p3, pn3 = load_npz(HERE / "photos_dinov3.npz")
c2raw = np.load(ROOT / "index" / "cad_embeddings.npy").astype(np.float32)
with open(ROOT / "index" / "cad_filenames.json", "r", encoding="utf-8") as f:
    n2 = json.load(f)
p2, pn2 = load_npz(HERE / "photos_dinov2.npz")

pos2 = {n: i for i, n in enumerate(n2)}
common = [n for n in n3 if n in pos2 and n not in excl]
keep3 = [i for i, n in enumerate(n3) if n in pos2 and n not in excl]
idx2 = [pos2[n] for n in common]
cad3 = project(CKPT3, c3[keep3])
cad2 = project(CKPT2, c2raw[idx2])
base2 = c2raw[idx2]                      # ham dinov2 (ikiz tekilleştirme için)
name_pos = {n: i for i, n in enumerate(common)}

# foto embeddingleri: 4. tur fotoğrafları photos_dinov2.npz'de olmayabilir —
# yalnız her iki tarafta da bulunanlar ölçülür (val/test zaten eski fotolar)
ph3 = dict(zip(pn3, p3))
ph2 = dict(zip(pn2, p2))


def prep(photos, ckpt3=CKPT3, ckpt2=CKPT2):
    usable = [(p, {Path(c).name for c in eslesme[p] if Path(c).name in name_pos})
              for p in photos if p in eslesme and p in ph3 and p in ph2]
    usable = [(p, t) for p, t in usable if t]
    q3 = project(ckpt3, np.stack([ph3[p] for p, _ in usable]))
    q2 = project(ckpt2, np.stack([ph2[p] for p, _ in usable]))
    return usable, q3, q2


def eval_w(usable, q3, q2, w, grouped=False):
    flat = {k: 0 for k in K_VALUES}
    card = {k: 0 for k in K_VALUES}
    for (p, targets), v3, v2 in zip(usable, q3, q2):
        scores = w * (cad3 @ v3) + (1 - w) * (cad2 @ v2)
        top = np.argpartition(-scores, 60)[:60]
        top = top[np.argsort(-scores[top])]
        tid = {name_pos[t] for t in targets}
        rank = next((r for r, i in enumerate(top[:20], 1) if int(i) in tid), None)
        for k in K_VALUES:
            if rank is not None and rank <= k:
                flat[k] += 1
        if grouped:
            reps = []  # (üye kümesi, temsilci vektörü)
            crank = None
            for i in top:
                v = base2[i]
                placed = False
                for gi, (members, rv) in enumerate(reps):
                    if float(rv @ v) >= DUP_SIM:
                        members.add(int(i))
                        placed = True
                        break
                if not placed and len(reps) < 20:
                    reps.append(({int(i)}, v))
            for gi, (members, _rv) in enumerate(reps, 1):
                if members & tid:
                    crank = gi
                    break
            for k in K_VALUES:
                if crank is not None and crank <= k:
                    card[k] += 1
    n = len(usable)
    return ({k: flat[k] / n for k in K_VALUES},
            {k: card[k] / n for k in K_VALUES}, n)


usable_v, q3v, q2v = prep(split["val"])
print("--- VAL w taraması")
best_w, best_r = None, -1
for w in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0):
    r, _, n = eval_w(usable_v, q3v, q2v, w)
    print(f"w={w:.2f}  " + "  ".join(f"r@{k} {r[k]:.1%}" for k in K_VALUES))
    if r[10] > best_r:
        best_r, best_w = r[10], w

usable_t, q3t, q2t = prep(split["test"])
rf, rc, n = eval_w(usable_t, q3t, q2t, best_w, grouped=True)
print(f"\nen iyi w={best_w} -> TEST ({n} foto):")
print("düz     " + "  ".join(f"r@{k} {rf[k]:.1%}" for k in K_VALUES))
print("gruplu  " + "  ".join(f"@{k} {rc[k]:.1%}" for k in K_VALUES))
