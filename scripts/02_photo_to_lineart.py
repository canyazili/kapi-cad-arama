# -*- coding: utf-8 -*-
"""
02_photo_to_lineart.py — Katalog fotoğraflarını lineart'a çevirir (toplu CLI).

Tek-görsel işleme mantığı (crop_door, remove_text, to_lineart, process_photo, ...)
artık kök modül photo_lineart.py'de; search.py de oradan import eder. Bu script
yalnızca toplu-işleme komut satırıdır (paketlenmiş exe scripts/ içermez).

Aşamalar (photo_lineart.process_photo):
  1) Kapı-crop: en büyük DİKEY içerik bloğunu bulur (kenar yazıları/logolar dışarıda).
  2) Metin silme (opsiyonel, config: text_removal.remove_text): easyocr + cv2.inpaint.
  3) controlnet_aux HEDdetector ile lineart; invert (beyaz zemin + koyu çizgi) + kare 518x518.

Kullanım:
  python scripts/02_photo_to_lineart.py            # tüm veri
  python scripts/02_photo_to_lineart.py --limit 30 # rastgele 30 dosya + data/debug kontrol görselleri
"""
import argparse
import random
import sys
import traceback
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Paket olmayan bu script doğrudan çalıştırıldığında proje kökü sys.path'te
# olmayabilir; photo_lineart/search'ü bulabilmek için ekle (01_clean_cad ile aynı desen).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Deney/araç scriptleri bu modülü importlib ile yükleyip aşağıdaki adları m.<ad>
# olarak kullanıyor; hepsini photo_lineart'tan yeniden dışa aktar (geriye uyumluluk).
from photo_lineart import (  # noqa: E402,F401
    crop_door, enhance_contrast, get_hed, get_ocr, ink_ratio, load_config,
    make_debug_image, process_photo, remove_text, to_lineart,
)


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
