# -*- coding: utf-8 -*-
"""
dev_fuzyon_olc.py — Dürüst (donmuş ayrımlı) modellerle FÜZYON ölçümü.

projected_final* üretim modelleri tüm veriyle eğitildiğinden dürüstçe
ölçülemez; bu skript verilen iki DEV checkpoint'iyle (dinov2 + dinov3)
üretimdeki 0.5/0.5 füzyonun aynısını kurar ve donmuş test kümesinde
recall@k ölçer. Foto başına sıraları da kaydeder (eski/yeni kıyası için).

Kullanım:
  python scripts/experiments/dev_fuzyon_olc.py --out eski.json
  python scripts/experiments/dev_fuzyon_olc.py --d3-ckpt projection_dev_dinov3_r5.pt --out yeni.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("train07", ROOT / "scripts" / "07_train_projection.py")
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)

from search import load_config, load_excluded_trivial  # noqa: E402


def side_setup(backbone, ckpt_name, cfg, device):
    """(common_names, cad_proj) — checkpoint'in MLP'siyle projekte CAD tarafı."""
    import torch
    bk = t7.BACKBONES[backbone]
    common, _ = t7.aligned_names(cfg, (bk["clean"], bk["hed"]))
    excluded = load_excluded_trivial(cfg) if cfg.get("exclude_trivial") else set()
    if excluded:
        common = [n for n in common if n not in excluded]
    clean = t7.load_side(cfg, bk["clean"], common)
    ckpt = torch.load(ROOT / "data/models" / ckpt_name, map_location="cpu")
    model = t7.make_model().to(device)
    model.load_state_dict(ckpt["state_dict"])
    cad_proj = t7.project_np(model, clean, device)
    return common, cad_proj, model, bk


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-ckpt", default="projection.pt")
    ap.add_argument("--d3-ckpt", default="projection_dev_dinov3_r4.pt")
    ap.add_argument("--photos-file", default=str(ROOT / "data/eval/projection_test_photos.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config()
    photos = json.load(open(args.photos_file, encoding="utf-8"))
    labels = json.load(open(ROOT / "data/eval/labels_clean.json", encoding="utf-8"))["eslesme"]

    c2, cad2, m2, bk2 = side_setup("dinov2", args.d2_ckpt, cfg, device)
    c3, cad3, m3, bk3 = side_setup("dinov3", args.d3_ckpt, cfg, device)
    names = sorted(set(c2) & set(c3))
    p2 = {n: i for i, n in enumerate(c2)}
    p3 = {n: i for i, n in enumerate(c3)}
    cad2 = cad2[[p2[n] for n in names]]
    cad3 = cad3[[p3[n] for n in names]]
    print(f"ortak CAD: {len(names)}", flush=True)

    cad_set = set(names)
    usable = {p: [Path(c).name for c in labels.get(p, []) if Path(c).name in cad_set]
              for p in photos}
    usable = {p: c for p, c in usable.items() if c}
    kept = sorted(usable)
    print(f"test fotosu: {len(photos)}, hedefi katalogda olan: {len(kept)}", flush=True)

    e2 = t7.collect_photo_embeddings_bk(bk2, kept, cfg)
    e3 = t7.collect_photo_embeddings_bk(bk3, kept, cfg)
    kept = [p for p in kept if p in e2 and p in e3]

    q2 = t7.project_np(m2, np.stack([e2[p] for p in kept]), device)
    q3 = t7.project_np(m3, np.stack([e3[p] for p in kept]), device)

    pos = {n: i for i, n in enumerate(names)}
    ranks, hits = {}, {k: 0 for k in (1, 5, 10, 20)}
    for i, p in enumerate(kept):
        scores = 0.5 * (cad2 @ q2[i]) + 0.5 * (cad3 @ q3[i])
        order = np.argsort(-scores)
        tset = {pos[c] for c in usable[p]}
        rank = next((r for r, j in enumerate(order, 1) if int(j) in tset), None)
        ranks[p] = rank
        for k in hits:
            if rank is not None and rank <= k:
                hits[k] += 1
    n = len(kept)
    recall = {f"r@{k}": round(hits[k] / n, 4) for k in sorted(hits)}
    print("FÜZYON", recall, f"(n={n})", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"d2": args.d2_ckpt, "d3": args.d3_ckpt, "n": n,
                   "recall": recall, "ranks": ranks}, f, ensure_ascii=False, indent=1)
    print("->", args.out, flush=True)


if __name__ == "__main__":
    main()
