# -*- coding: utf-8 -*-
"""
gruplu_kart_olc.py — Dev füzyon skorlarıyla GRUPLU KART isabeti (uygulama görünümü).

dev_fuzyon_olc ile aynı füzyonu kurar; her test fotosu için top-60 ham sonucu
kapi_arama_app.group_results ile ikiz-tekilleştirir ve doğru tasarımın kaçıncı
KARTTA geldiğini ölçer (kart = temsilci + varyantları; herhangi biri hedefse isabet).

Kullanım: python scripts/experiments/gruplu_kart_olc.py [--d2-ckpt ...] [--d3-ckpt ...]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"c:/Users/canya/Desktop/kapı")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("dfo", ROOT / "scripts/experiments/dev_fuzyon_olc.py")
dfo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo)
t7 = dfo.t7

from search import load_config  # noqa: E402
import kapi_arama_app as app  # noqa: E402


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-ckpt", default="projection.pt")
    ap.add_argument("--d3-ckpt", default="projection_dev_dinov3_r5.pt")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config()

    photos = json.load(open(ROOT / "data/eval/projection_test_photos.json", encoding="utf-8"))
    labels = json.load(open(ROOT / "data/eval/labels_clean.json", encoding="utf-8"))["eslesme"]

    c2, cad2, m2, bk2 = dfo.side_setup("dinov2", args.d2_ckpt, cfg, device)
    c3, cad3, m3, bk3 = dfo.side_setup("dinov3", args.d3_ckpt, cfg, device)
    names = sorted(set(c2) & set(c3))
    p2 = {n: i for i, n in enumerate(c2)}
    p3 = {n: i for i, n in enumerate(c3)}
    cad2 = cad2[[p2[n] for n in names]]
    cad3 = cad3[[p3[n] for n in names]]

    cad_set = set(names)
    usable = {p: {Path(c).name for c in labels.get(p, []) if Path(c).name in cad_set}
              for p in photos}
    usable = {p: c for p, c in usable.items() if c}
    kept = sorted(usable)

    e2 = t7.collect_photo_embeddings_bk(bk2, kept, cfg)
    e3 = t7.collect_photo_embeddings_bk(bk3, kept, cfg)
    kept = [p for p in kept if p in e2 and p in e3]
    q2 = t7.project_np(m2, np.stack([e2[p] for p in kept]), device)
    q3 = t7.project_np(m3, np.stack([e3[p] for p in kept]), device)

    hits = {k: 0 for k in (1, 5, 10, 20)}
    for i, p in enumerate(kept):
        scores = 0.5 * (cad2 @ q2[i]) + 0.5 * (cad3 @ q3[i])
        top = np.argpartition(-scores, args.k * 3)[: args.k * 3]
        top = top[np.argsort(-scores[top])]
        raw = [(names[int(j)], float(scores[int(j)])) for j in top]
        cards = app.group_results(raw, args.k)
        rank = None
        for r, (nm, _s, varyant) in enumerate(cards, 1):
            if nm in usable[p] or usable[p] & {v[0] for v in varyant}:
                rank = r
                break
        for k in hits:
            if rank is not None and rank <= k:
                hits[k] += 1
    n = len(kept)
    print(f"GRUPLU KART (n={n}, d2={args.d2_ckpt}, d3={args.d3_ckpt}):")
    print("  " + "  ".join(f"@{k} {hits[k]/n:.1%}" for k in sorted(hits)))


if __name__ == "__main__":
    main()
