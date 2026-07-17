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

import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# normalize_cad + sabitler artık kök modülde (paketlenmiş exe scripts/ içermediği için).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cad_normalize import normalize_cad, make_debug_image  # noqa: E402


def load_config():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
