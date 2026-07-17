# -*- coding: utf-8 -*-
"""
galeri_uret.py — TÜM fotoğraflar için işlenmiş SON HALİ (lineart) üretir.

Her foto için üretim boru hattı (02_photo_to_lineart.process_photo ile birebir,
CLAHE kurtarması dahil) çalıştırılır; aramada kullanılan çizgi çıkarımı
kaydedilir. Çıktı: data/debug/galeri/panel/<ad>.jpg + meta.jsonl (ink, bayraklar)

Yarıda kesilirse kaldığı yerden devam eder (mevcut paneller atlanır).
Kullanım: python scripts/experiments/galeri_uret.py [--limit N]
"""
import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
m = importlib.import_module("02_photo_to_lineart")

Image.MAX_IMAGE_PIXELS = None
PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp")
OUT_DIR = ROOT / "data" / "debug" / "galeri"
PANEL_DIR = OUT_DIR / "panel"
PANEL_H = 340
PANEL_Q = 72


def crop_door_bbox(img: Image.Image):
    """m.crop_door kopyası; kutu + fallback bilgisini de döndürür."""
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < cw:
            continue
        if cw < m.MIN_ASPECT_RATIO * ch:
            continue
        if cw * ch < m.MIN_AREA_RATIO * w * h or ch < m.MIN_HEIGHT_RATIO * h:
            continue
        if best is None or cw * ch > best[2] * best[3]:
            best = (x, y, cw, ch)
    if best is None:
        band_w = min(w, int(h * m.FALLBACK_BAND_RATIO))
        x0 = (w - band_w) // 2
        box = (x0, 0, x0 + band_w, h)
        return img.crop(box), box, True
    x, y, cw, ch = best
    mg = int(max(cw, ch) * m.CROP_MARGIN_RATIO)
    box = (max(x - mg, 0), max(y - mg, 0), min(x + cw + mg, w), min(y + ch + mg, h))
    return img.crop(box), box, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = m.load_config()
    target = cfg["image_size"]
    do_clean = cfg["text_removal"]["remove_text"]

    photos = sorted(p for p in (ROOT / "photos").iterdir()
                    if p.suffix.lower() in PHOTO_EXTS)
    if args.limit:
        photos = photos[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "meta.jsonl"
    done = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["foto"])
                except Exception:
                    pass
    todo = [p for p in photos if p.name not in done]
    print(f"Toplam {len(photos)} foto, {len(done)} hazır, {len(todo)} işlenecek",
          flush=True)
    m.get_hed()
    if do_clean:
        m.get_ocr()

    t0 = time.time()
    meta_f = open(meta_path, "a", encoding="utf-8")
    for i, f in enumerate(todo):
        rec = {"foto": f.name}
        try:
            img = Image.open(f)
            img.load()
            if img.mode in ("RGBA", "LA", "PA"):
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img.convert("RGBA"))
            img = img.convert("RGB")
            W, H = img.size

            # --- process_photo akışının birebir kopyası (bayraklar için açık) ---
            cropped, box, fb = crop_door_bbox(img)
            cleaned = m.remove_text(cropped, cfg) if do_clean else None
            lineart = m.to_lineart(cleaned if cleaned is not None else cropped, target)
            ink = m.ink_ratio(lineart)
            clahe_used = tam_kare = False
            if ink < m.MIN_INK_RATIO:                      # Kurtarma 1: kontrast
                src = cleaned if cleaned is not None else cropped
                la2 = m.to_lineart(m.enhance_contrast(src), target)
                if m.ink_ratio(la2) > ink:
                    lineart, ink, clahe_used = la2, m.ink_ratio(la2), True
            if ink < m.MIN_INK_RATIO and cropped.size != img.size:  # Kurtarma 2
                cl_full = m.remove_text(img, cfg) if do_clean else None
                la_full = m.to_lineart(cl_full if cl_full is not None else img, target)
                if m.ink_ratio(la_full) > ink:
                    lineart, cropped, cleaned = la_full, img, cl_full
                    ink, tam_kare, box = m.ink_ratio(la_full), True, (0, 0, W, H)

            # sadece son hal: aramada kullanılan çizgi çıkarımı
            panel = lineart.resize(
                (max(1, int(lineart.width * PANEL_H / lineart.height)), PANEL_H))
            # tam ad + .jpg: "X.jpg" ile "X.jpeg" çakışmasın
            panel.convert("RGB").save(PANEL_DIR / (f.name + ".jpg"), quality=PANEL_Q)

            flags = []
            if fb:
                flags.append("fallback_bant")
            if clahe_used:
                flags.append("kontrast_kurtarma")
            if tam_kare:
                flags.append("tam_kare")
            if ink < m.MIN_INK_RATIO:
                flags.append("zayif_lineart")
            crop_ratio = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
            rec.update({"ink": round(ink, 4), "crop_ratio": round(crop_ratio, 3),
                        "bayrak": flags})
        except Exception:
            rec["hata"] = traceback.format_exc(limit=1).strip().splitlines()[-1][:200]
        meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            meta_f.flush()
            el = time.time() - t0
            print(f"{i+1}/{len(todo)}  {el/60:.1f} dk  "
                  f"(~{el/(i+1)*len(todo)/60:.0f} dk toplam)", flush=True)
    meta_f.close()
    print(f"BITTI: {len(todo)} foto, {(time.time()-t0)/60:.1f} dk", flush=True)


if __name__ == "__main__":
    main()
