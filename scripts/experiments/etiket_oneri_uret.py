# -*- coding: utf-8 -*-
"""
etiket_oneri_uret.py — Tek-tek etiketleme sayfası için veri hazırlığı.

Etiketsiz her fotoğraf için:
  - önbellekteki lineart ile füzyon motorundan top-K öneri (skorlu)
  - sayfada hızlı açılsın diye küçük önizleme (thumbs/, uzun kenar 900px)
Çıktı: data/eval/etiketleme/oneriler.json + thumbs/*.jpg

Kullanım: python scripts/experiments/etiket_oneri_uret.py [--top-k 12]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(r"c:/Users/canya/Desktop/kapı")
sys.path.insert(0, str(ROOT))

Image.MAX_IMAGE_PIXELS = None
PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp")
OUT_DIR = ROOT / "data" / "eval" / "etiketleme"
THUMB_DIR = OUT_DIR / "thumbs"
CACHE_DIR = ROOT / "data" / "eval" / "cache"
THUMB_MAX = 900


def make_thumb(src: Path, dst: Path):
    if dst.exists():
        return
    img = Image.open(src)
    # dev JPEG'lerde draft ile hızlı küçültme
    if img.format == "JPEG":
        img.draft("RGB", (THUMB_MAX * 2, THUMB_MAX * 2))
    img.load()
    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    img = img.convert("RGB")
    img.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
    img.save(dst, "JPEG", quality=82)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()

    labels = json.load(open(ROOT / "data/eval/labels_clean.json", encoding="utf-8"))
    labeled = set(labels["eslesme"])
    photos = sorted(p for p in (ROOT / "photos").iterdir()
                    if p.suffix.lower() in PHOTO_EXTS and p.name not in labeled)
    print(f"Etiketsiz foto: {len(photos)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "oneriler.json"
    cands = {}
    if out_path.exists():
        cands = json.load(open(out_path, encoding="utf-8"))

    from search import get_engine
    engine = get_engine()

    t0 = time.time()
    errors = 0
    for i, p in enumerate(photos):
        try:
            # tam ad + .jpg: "X.jpg" ile "X.jpeg" ayrı önizleme alsın (çakışma olmasın)
            make_thumb(p, THUMB_DIR / (p.name + ".jpg"))
            if p.name in cands and isinstance(cands[p.name], list):
                continue
            cp = CACHE_DIR / (p.name + ".lineart.png")
            if cp.exists():
                with Image.open(cp) as im:
                    im.load()
                    lineart = im.convert("RGB")
            else:
                lineart, _, _ = engine.prepare_query(p)
                lineart.save(cp)
            results = engine.search_prepared(lineart, k=args.top_k)
            cands[p.name] = [[n, round(float(s), 4)] for n, s in results]
        except Exception as e:
            errors += 1
            cands[p.name] = {"hata": str(e)[:200]}
        if (i + 1) % 50 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cands, f, ensure_ascii=False)
            el = time.time() - t0
            print(f"{i+1}/{len(photos)}  {el/60:.1f} dk", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)
    print(f"BITTI: {len(cands)} foto, {errors} hata, "
          f"{(time.time()-t0)/60:.1f} dk. Çıktı: {out_path}", flush=True)


if __name__ == "__main__":
    main()
