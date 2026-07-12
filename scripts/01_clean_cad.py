# -*- coding: utf-8 -*-
"""
01_clean_cad.py — CAD PNG normalizasyonu.

cad_png klasöründeki AutoCAD export PNG'lerini standart hale getirir:
  - Şeffaf (RGBA) zemini beyaza composite eder
  - Zemin koyuysa (siyah zemin + beyaz çizgi) invert eder -> beyaz zemin + koyu çizgi
  - İçeriğin bounding box'ını bulup kırpar
  - Yatay export edilmiş kapıları 90 derece döndürür (hepsi dikey dursun)
  - En-boy oranını BOZMADAN beyaz zeminle kare padding + 518x518 resize
Çıktı: data/cad_clean. Bozuk dosyalar atlanır ve data/errors.log'a yazılır.

Kullanım:
  python scripts/01_clean_cad.py            # tüm veri
  python scripts/01_clean_cad.py --limit 50 # rastgele 50 dosya + data/debug kontrol görselleri
"""
import argparse
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# Koyu piksel eşiği: bbox tespiti bu değerin altındaki pikselleri "içerik" sayar
DARK_THRESHOLD = 200
# Zemin koyu mu kararı: ortalama parlaklık bunun altındaysa invert edilir
INVERT_MEAN_THRESHOLD = 128
# Crop sonrası içeriğin etrafına bırakılan pay (uzun kenarın oranı)
BBOX_MARGIN_RATIO = 0.02


def load_config():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_cad(img: Image.Image, target_size: int) -> Image.Image:
    """Tek bir CAD görselini standart forma getirir (beyaz zemin, koyu çizgi, dikey, kare)."""
    # 1) Şeffaflık gerçekten kullanılmışsa çizgi rengine hiç bakma, doğrudan
    #    alpha kanalını kullan: opak piksel (çizgi) koyu, şeffaf zemin beyaz olur.
    #    Böylece şeffaf zemin üzerine BEYAZ çizgili exportlar da doğru çıkar
    #    (beyaza composite edilince çizginin kaybolduğu hata bu yolla çözülür).
    arr = None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        if alpha.min() < 255:  # şeffaflık gerçekten var
            arr = 255 - alpha
        else:  # alpha tamamen opak: şeffaflık bilgisi yok, normal yoldan devam
            img = rgba

    if arr is None:
        # 2) Alpha yolu kullanılmadıysa: griye çevir, zemin koyuysa invert
        #    -> hedef: beyaz zemin üzerine koyu çizgi
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        if arr.mean() < INVERT_MEAN_THRESHOLD:
            arr = 255 - arr

    # 3) Koyu piksellere göre içerik bbox'ı
    if arr.min() == arr.max():
        raise ValueError("görsel tek renk — boş/bozuk export, yeniden export gerekli")
    mask = arr < DARK_THRESHOLD
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("içerik bulunamadı (görsel tamamen boş/beyaz)")
    margin = int(max(arr.shape) * BBOX_MARGIN_RATIO)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, arr.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, arr.shape[1])
    arr = arr[y0:y1, x0:x1]

    # 4) Yatay export edilmişse dikeye döndür
    h, w = arr.shape
    if w > h:
        arr = np.rot90(arr)  # 90 derece, oran korunur
        h, w = arr.shape

    # 5) Beyaz zeminle kare padding (oran bozulmaz), sonra resize
    side = max(h, w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = arr

    out = Image.fromarray(canvas).resize((target_size, target_size), Image.LANCZOS)
    return out.convert("RGB")


def make_debug_image(before: Image.Image, after: Image.Image) -> Image.Image:
    """Önce/sonra halini yan yana gösteren kontrol görseli üretir."""
    h = 400
    b = before.convert("RGB")
    b = b.resize((max(1, int(b.width * h / b.height)), h))
    a = after.convert("RGB").resize((h, h))
    canvas = Image.new("RGB", (b.width + a.width + 10, h), (255, 0, 0))
    canvas.paste(b, (0, 0))
    canvas.paste(a, (b.width + 10, 0))
    return canvas


def process_one(src: str, dst: str, target_size: int, debug_dir: str | None):
    """Worker: tek dosya işler. (dosya_adı, hata_mesajı|None) döner."""
    src_p, dst_p = Path(src), Path(dst)
    try:
        with Image.open(src_p) as im:
            im.load()
            before = im.copy() if debug_dir else None
            out = normalize_cad(im, target_size)
        out.save(dst_p)
        if debug_dir:
            make_debug_image(before, out).save(Path(debug_dir) / src_p.name)
        return src_p.name, None
    except Exception:
        return src_p.name, traceback.format_exc(limit=1).strip().splitlines()[-1]


def main():
    parser = argparse.ArgumentParser(description="CAD PNG normalizasyonu")
    parser.add_argument("--limit", type=int, default=None,
                        help="Rastgele N dosyayla test; debug görselleri data/debug/cad'e yazılır")
    parser.add_argument("--workers", type=int, default=None, help="Paralel işçi sayısı")
    parser.add_argument("--only-missing", action="store_true",
                        help="Sadece cad_png'de olup data/cad_clean'de OLMAYAN dosyaları işle "
                             "(daha önce atlananları yeniden denemek için)")
    args = parser.parse_args()

    cfg = load_config()
    src_dir = ROOT / cfg["paths"]["cad_png"]
    dst_dir = ROOT / cfg["paths"]["cad_clean"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    errors_log = ROOT / cfg["paths"]["errors_log"]
    target_size = cfg["image_size"]

    files = sorted(src_dir.glob("*.png"))
    if not files:
        print(f"UYARI: {src_dir} içinde PNG bulunamadı.", file=sys.stderr)
        sys.exit(1)

    if args.only_missing:
        total = len(files)
        files = [f for f in files if not (dst_dir / f.name).exists()]
        print(f"--only-missing: {total} dosyanın {len(files)} tanesi eksik, onlar işlenecek.")
        if not files:
            print("Eksik dosya yok, çıkılıyor.")
            return

    debug_dir = None
    if args.limit is not None:
        random.seed(42)
        files = random.sample(files, min(args.limit, len(files)))
        debug_dir = ROOT / cfg["paths"]["debug"] / "cad"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"--limit modu: {len(files)} dosya, kontrol görselleri -> {debug_dir}")

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_one, str(f), str(dst_dir / f.name), target_size,
                        str(debug_dir) if debug_dir else None)
            for f in files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="CAD normalizasyon", unit="img"):
            name, err = fut.result()
            if err:
                errors.append(f"{name}\t{err}")

    if errors:
        with open(errors_log, "a", encoding="utf-8") as f:
            f.write("\n".join(errors) + "\n")
        print(f"{len(errors)} dosya atlandı, detay: {errors_log}")
    print(f"Bitti: {len(files) - len(errors)}/{len(files)} dosya -> {dst_dir}")


if __name__ == "__main__":
    main()
