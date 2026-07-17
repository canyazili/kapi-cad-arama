# -*- coding: utf-8 -*-
"""
audit_crop_text.py — Etiketli fotoğraflarda kırpma + yazı silme denetimi.

Her etiketli foto için üretim boru hattını (02_photo_to_lineart.process_photo
akışının birebir kopyası, kırpma kutusunu da raporlayacak şekilde) çalıştırır:
  - kırpma kutusu bulundu mu, fallback bandı mı kullanıldı?
  - ink-ratio güvenlik ağı tam kareye düşürdü mü?
  - OCR (tam görüntüde) bulunan yazılar: kırpmayla dışarıda mı kaldı,
    inpaint ile silindi mi (conf>=0.5), yoksa hayatta mı (conf<0.5)?
Şüpheli vakalar için debug paneli (orijinal|crop|temiz|lineart) kaydeder ve
sonunda data/debug/kirpma_denetim/rapor.json + inceleme sayfası üretir.

Kullanım: python scripts/experiments/audit_crop_text.py [--limit N]
"""
import argparse
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
import importlib
m = importlib.import_module("02_photo_to_lineart")

Image.MAX_IMAGE_PIXELS = None
OUT_DIR = ROOT / "data" / "debug" / "kirpma_denetim"
PANEL_DIR = OUT_DIR / "panel"


def crop_door_bbox(img: Image.Image):
    """m.crop_door'un birebir kopyası; kutuyu da döndürür.
    Dönüş: (cropped, bbox, fallback_mu)  bbox=(x0,y0,x1,y1) orijinal koordinatta."""
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


def box_inside_frac(poly, box):
    """OCR poligon köşelerinin kırpma kutusu içinde kalan oranı."""
    x0, y0, x1, y1 = box
    pts = np.asarray(poly, dtype=np.float32)
    inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
              (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
    return float(inside.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = m.load_config()
    target = cfg["image_size"]
    min_conf = cfg["text_removal"]["min_confidence"]
    labels = json.load(open(ROOT / "data/eval/labels_clean.json", encoding="utf-8"))
    photos = sorted(labels["eslesme"].keys())
    if args.limit:
        photos = photos[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    m.get_hed()
    ocr = m.get_ocr()

    rows, errors = [], []
    t0 = time.time()
    for i, name in enumerate(photos):
        f = ROOT / "photos" / name
        try:
            img = Image.open(f)
            img.load()
            if img.mode in ("RGBA", "LA", "PA"):
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img.convert("RGBA"))
            img = img.convert("RGB")
            W, H = img.size

            cropped, box, fb = crop_door_bbox(img)
            cleaned = m.remove_text(cropped, cfg)
            lineart = m.to_lineart(cleaned, target)
            ink = m.ink_ratio(lineart)
            used_full = False
            if ink < m.MIN_INK_RATIO and cropped.size != img.size:
                cleaned_full = m.remove_text(img, cfg)
                lineart_full = m.to_lineart(cleaned_full, target)
                if m.ink_ratio(lineart_full) > ink:
                    lineart, cropped, cleaned = lineart_full, img, cleaned_full
                    box, used_full = (0, 0, W, H), True
                    ink = m.ink_ratio(lineart)

            # Tam görüntüde OCR: yazılar nereye düştü?
            dets = ocr.readtext(np.asarray(img))
            txt_out, txt_removed, txt_alive = [], [], []
            for poly, text, conf in dets:
                t = text.strip()
                if len(t) < 2:
                    continue
                fr = box_inside_frac(poly, box)
                item = [t, round(float(conf), 3)]
                if fr < 0.5:
                    txt_out.append(item)       # kırpma dışarıda bıraktı
                elif conf >= min_conf:
                    txt_removed.append(item)   # inpaint sildi
                else:
                    txt_alive.append(item)     # crop içinde, silinmedi

            crop_ratio = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
            flags = []
            if fb:
                flags.append("fallback_bant")
            if used_full:
                flags.append("tam_kare")
            if ink < m.MIN_INK_RATIO:
                flags.append("bos_lineart")
            if txt_alive:
                flags.append("yazi_hayatta")
            if crop_ratio > 0.97 and (txt_removed or txt_alive or txt_out):
                flags.append("kirpma_yok_yazili")

            rows.append({
                "foto": name, "boyut": [W, H], "bbox": list(box),
                "crop_ratio": round(crop_ratio, 3), "fallback": fb,
                "tam_kare": used_full, "ink": round(ink, 4),
                "yazi_disarida": txt_out, "yazi_silindi": txt_removed,
                "yazi_hayatta": txt_alive, "bayrak": flags,
            })
            if flags:
                dbg = m.make_debug_image(img, cropped, cleaned, lineart)
                dbg.save(PANEL_DIR / (Path(name).stem + ".jpg"), quality=80)
        except Exception:
            err = traceback.format_exc(limit=1).strip().splitlines()[-1]
            errors.append({"foto": name, "hata": err})

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(photos)}  {el/60:.1f} dk  (~{el/(i+1)*len(photos)/60:.0f} dk toplam)",
                  flush=True)

    out = {"toplam": len(photos), "hata": errors, "sonuc": rows}
    with open(OUT_DIR / "rapor.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    # Özet
    n = len(rows)
    fbn = sum(r["fallback"] for r in rows)
    tkn = sum(r["tam_kare"] for r in rows)
    bosn = sum("bos_lineart" in r["bayrak"] for r in rows)
    aliven = sum(bool(r["yazi_hayatta"]) for r in rows)
    outn = sum(bool(r["yazi_disarida"]) for r in rows)
    remn = sum(bool(r["yazi_silindi"]) for r in rows)
    print(f"\n=== ÖZET ({n} foto, {len(errors)} hata) ===")
    print(f"fallback bant kullanılan : {fbn}")
    print(f"ink güvenlik ağı tam kare: {tkn}")
    print(f"boş/zayıf lineart        : {bosn}")
    print(f"yazı kırpmayla dışarıda  : {outn}")
    print(f"yazı inpaint ile silinen : {remn}")
    print(f"yazı HAYATTA (sorunlu)   : {aliven}")
    print(f"panel klasörü            : {PANEL_DIR}")


if __name__ == "__main__":
    main()
