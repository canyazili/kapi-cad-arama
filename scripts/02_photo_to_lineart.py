# -*- coding: utf-8 -*-
"""
02_photo_to_lineart.py — Katalog fotoğraflarını lineart'a çevirir.

Aşamalar:
  1) Kapı-crop: kenar yoğunluğu / kontur analiziyle görüntüdeki en büyük DİKEY
     dikdörtgen içerik bloğunu bulur; kenardaki model yazıları ve logolar dışarıda
     kalır. Bulunamazsa fallback: görüntünün ortasındaki dikey bant.
  2) Metin temizleme (opsiyonel, config: text_removal.remove_text): easyocr
     (TR+EN) ile metin kutuları bulunur, güven skoru eşiğin üstündekiler
     cv2.inpaint (TELEA) ile çevre dokuya göre doldurulur. Crop içinde kalan
     model adı / logo yazıları embedding'i kirletmesin diye.
  3) controlnet_aux HEDdetector ile lineart (CPU'da da çalışır). HED çıktısı
     invert edilir ki CAD tarafıyla aynı forma gelsin: beyaz zemin + koyu çizgi.
Son adım: kare padding + 518x518.

Bu modül search.py tarafından import edilir: process_photo(img) tek görsel işler.

Kullanım:
  python scripts/02_photo_to_lineart.py            # tüm veri
  python scripts/02_photo_to_lineart.py --limit 30 # rastgele 30 dosya + data/debug kontrol görselleri
"""
import argparse
import random
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# Bazı katalog fotoğrafları 180M+ piksel; PIL'in decompression-bomb sınırına
# takılıyor. Kaynak yerel ve güvenilir olduğundan sınır kaldırıldı.
Image.MAX_IMAGE_PIXELS = None

# Kapı-crop parametreleri
MIN_AREA_RATIO = 0.08      # aday blok, görüntü alanının en az bu oranı olmalı
MIN_HEIGHT_RATIO = 0.40    # aday blok, görüntü yüksekliğinin en az bu oranı olmalı
MIN_ASPECT_RATIO = 0.25    # aday blok genişliği, yüksekliğinin en az bu katı olmalı
                           # (daha darı kapı olamaz — kol/pervaz şeridi yakalanmasın)
CROP_MARGIN_RATIO = 0.02   # bulunan bloğun etrafına bırakılan pay
FALLBACK_BAND_RATIO = 0.6  # fallback dikey bant genişliği (yüksekliğe oran)
MIN_INK_RATIO = 0.012      # lineart'ta koyu piksel oranı bunun altındaysa kırpma
                           # şüpheli sayılır, tam kareyle yeniden denenir (test
                           # kümesi ölçümü: sağlıklı fotoların p10'u ~0.018)

_hed = None  # HEDdetector tembel yüklenir (import eden herkes model indirmesin)
_ocr = None  # easyocr Reader da tembel yüklenir


def load_config():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_hed():
    """HEDdetector'ı bir kez yükler ve önbelleğe alır (CPU'da da çalışır)."""
    global _hed
    if _hed is None:
        from controlnet_aux import HEDdetector
        cfg = load_config()
        _hed = HEDdetector.from_pretrained(cfg["hed"]["pretrained"])
    return _hed


def get_ocr():
    """easyocr Reader'ı (Türkçe + İngilizce) bir kez yükler ve önbelleğe alır."""
    global _ocr
    if _ocr is None:
        import easyocr
        import torch
        _ocr = easyocr.Reader(["tr", "en"], gpu=torch.cuda.is_available(), verbose=False)
    return _ocr


def remove_text(img: Image.Image, cfg: dict = None) -> Image.Image:
    """Görüntüdeki metin bölgelerini OCR ile bulup inpaint (TELEA) ile siler.

    Sadece güven skoru text_removal.min_confidence üstündeki tespitler silinir;
    böylece kapı deseni yanlışlıkla metin sanılıp bozulmaz.
    """
    tr = (cfg or load_config())["text_removal"]
    rgb = np.asarray(img.convert("RGB"))
    detections = get_ocr().readtext(rgb)

    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for box, _text, conf in detections:
        if conf >= tr["min_confidence"]:
            cv2.fillPoly(mask, [np.asarray(box, dtype=np.int32)], 255)
    if not mask.any():
        return img  # eşiği geçen metin yok, görüntüye dokunma

    d = int(tr.get("mask_dilate", 7))
    if d > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (d, d)))
    cleaned = cv2.inpaint(rgb, mask, float(tr.get("inpaint_radius", 3)), cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned)


def crop_door(img: Image.Image) -> Image.Image:
    """Görüntüdeki en büyük dikey içerik bloğunu (kapıyı) kırpar.

    Kenar haritası çıkarılıp genişletilir; dış konturlar arasından
    "dikey + yeterince büyük" olan en geniş alanlı blok seçilir.
    Aday yoksa görüntünün ortasındaki dikey bant döner.
    """
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Kenarları birbirine bağla ki kapı tek blok halinde yakalansın
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < cw:  # yatay blok (yazı şeridi vb.) isteme, kapı dikeydir
            continue
        if cw < MIN_ASPECT_RATIO * ch:  # aşırı dar dikey şerit (kol, pervaz) da kapı değil
            continue
        if cw * ch < MIN_AREA_RATIO * w * h or ch < MIN_HEIGHT_RATIO * h:
            continue
        if best is None or cw * ch > best[2] * best[3]:
            best = (x, y, cw, ch)

    if best is None:
        # Fallback: ortadaki dikey bant
        band_w = min(w, int(h * FALLBACK_BAND_RATIO))
        x0 = (w - band_w) // 2
        return img.crop((x0, 0, x0 + band_w, h))

    x, y, cw, ch = best
    m = int(max(cw, ch) * CROP_MARGIN_RATIO)
    return img.crop((max(x - m, 0), max(y - m, 0),
                     min(x + cw + m, w), min(y + ch + m, h)))


def to_lineart(img: Image.Image, target_size: int) -> Image.Image:
    """HED ile lineart üretir, invert eder (beyaz zemin + koyu çizgi), kare 518x518 yapar."""
    hed = get_hed()
    line = hed(img.convert("RGB"), detect_resolution=512, image_resolution=512)
    # HED çıktısı siyah zemin + beyaz kenar -> CAD ile aynı forma getirmek için invert
    arr = 255 - np.asarray(line.convert("L"), dtype=np.uint8)

    h, w = arr.shape
    side = max(h, w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = arr
    out = Image.fromarray(canvas).resize((target_size, target_size), Image.LANCZOS)
    return out.convert("RGB")


def ink_ratio(lineart: Image.Image) -> float:
    """Lineart'ta koyu (çizgi) piksel oranı — boş/bozuk çıktı tespiti için."""
    g = np.asarray(lineart.convert("L"))
    return float((g < 128).mean())


def enhance_contrast(img: Image.Image) -> Image.Image:
    """CLAHE ile bölgesel kontrast açar (L kanalında). Koyu/soluk kapılarda
    HED'in göremediği desen çizgilerini görünür kılmak için; sadece çizgi
    oranı MIN_INK_RATIO altında kalan fotolarda devreye girer."""
    lab = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def process_photo(img_or_path, target_size: int = None, return_steps: bool = False):
    """Tek fotoğrafı arama pipeline'ına hazırlar: kapı-crop [+ metin silme] + HED lineart.

    Kırpma güvenliği: lineart'taki çizgi oranı MIN_INK_RATIO altındaysa kırpma
    büyük ihtimalle kapıyı kaçırmıştır (ör. sadece kol şeridi) — tam kareyle
    yeniden denenir ve çizgisi zengin olan sonuç kullanılır.

    img_or_path: PIL.Image veya dosya yolu.
    return_steps=True ise (lineart, cropped, cleaned) üçlüsü döner (debug görünümü
    için); metin silme kapalıysa cleaned None olur.
    """
    cfg = load_config()
    if target_size is None:
        target_size = cfg["image_size"]
    if isinstance(img_or_path, (str, Path)):
        img = Image.open(img_or_path)
        img.load()
    else:
        img = img_or_path
    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    img = img.convert("RGB")

    do_clean = cfg["text_removal"]["remove_text"]
    cropped = crop_door(img)
    cleaned = remove_text(cropped, cfg) if do_clean else None
    lineart = to_lineart(cleaned if cleaned is not None else cropped, target_size)

    # Kurtarma 1: çizgi çok azsa (koyu/soluk kapı) kontrastı açıp yeniden dene
    # (2026-07-14 denetimi: 39 fotoda desen çizgileri HED'de kayboluyordu)
    if ink_ratio(lineart) < MIN_INK_RATIO:
        src = cleaned if cleaned is not None else cropped
        lineart_enh = to_lineart(enhance_contrast(src), target_size)
        if ink_ratio(lineart_enh) > ink_ratio(lineart):
            lineart = lineart_enh

    # Kurtarma 2: hâlâ azsa kırpma kapıyı kaçırmış olabilir — tam kareyle dene
    if ink_ratio(lineart) < MIN_INK_RATIO and cropped.size != img.size:
        cleaned_full = remove_text(img, cfg) if do_clean else None
        lineart_full = to_lineart(cleaned_full if cleaned_full is not None else img,
                                  target_size)
        if ink_ratio(lineart_full) > ink_ratio(lineart):
            lineart, cropped, cleaned = lineart_full, img, cleaned_full

    if return_steps:
        return lineart, cropped, cleaned
    return lineart


def make_debug_image(*images: Image.Image) -> Image.Image:
    """Verilen adımları (orijinal | crop | metin-temizlenmiş | lineart) yan yana dizer.

    None olan adımlar (ör. metin silme kapalıyken cleaned) atlanır.
    """
    h = 400
    panels = []
    for p in images:
        if p is None:
            continue
        p = p.convert("RGB")
        panels.append(p.resize((max(1, int(p.width * h / p.height)), h)))
    total_w = sum(p.width for p in panels) + 10 * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, h), (255, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 10
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Fotoğraf -> kapı-crop -> HED lineart")
    parser.add_argument("--limit", type=int, default=None,
                        help="Rastgele N dosyayla test; debug görselleri data/debug/photos'a yazılır")
    args = parser.parse_args()

    cfg = load_config()
    src_dir = ROOT / cfg["paths"]["photos"]
    dst_dir = ROOT / cfg["paths"]["photos_lineart"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    errors_log = ROOT / cfg["paths"]["errors_log"]
    target_size = cfg["image_size"]

    files = sorted(p for p in src_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not files:
        print(f"UYARI: {src_dir} içinde görsel bulunamadı.", file=sys.stderr)
        sys.exit(1)

    debug_dir = None
    if args.limit is not None:
        random.seed(42)
        files = random.sample(files, min(args.limit, len(files)))
        debug_dir = ROOT / cfg["paths"]["debug"] / "photos"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"--limit modu: {len(files)} dosya, kontrol görselleri -> {debug_dir}")

    get_hed()  # modelleri baştan yükle ki progress bar temiz aksın
    if cfg["text_removal"]["remove_text"]:
        get_ocr()

    errors = []
    for f in tqdm(files, desc="Fotoğraf -> lineart", unit="img"):
        try:
            with Image.open(f) as im:
                im.load()
                original = im.copy() if debug_dir else None
            lineart, cropped, cleaned = process_photo(f, target_size, return_steps=True)
            lineart.save(dst_dir / (f.stem + ".png"))
            if debug_dir:
                make_debug_image(original, cropped, cleaned, lineart).save(
                    debug_dir / (f.stem + ".png"))
        except Exception:
            err = traceback.format_exc(limit=1).strip().splitlines()[-1]
            errors.append(f"{f.name}\t{err}")

    if errors:
        with open(errors_log, "a", encoding="utf-8") as fh:
            fh.write("\n".join(errors) + "\n")
        print(f"{len(errors)} dosya atlandı, detay: {errors_log}")
    print(f"Bitti: {len(files) - len(errors)}/{len(files)} dosya -> {dst_dir}")


if __name__ == "__main__":
    main()
