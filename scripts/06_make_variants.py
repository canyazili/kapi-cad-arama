# -*- coding: utf-8 -*-
"""
06_make_variants.py — CAD tarafı için veri varyantları üretir (domain gap deneyleri).

İki mod:
  thick: data/cad_clean -> data/variants/cad_thick<K>
         Koyu çizgilere KxK morfolojik kalınlaştırma (beyaz zeminde koyu çizgi
         için erode) + hafif Gaussian blur. Sorgu tarafındaki kalın/yumuşak HED
         fırça stiline ucuz bir yaklaşım. Multiprocessing ile hızlı.

  hed:   data/cad_clean -> data/variants/cad_hed
         CAD görsellerini de sorgu fotoğraflarıyla AYNI HED detektöründen geçirir
         (aynı invert dahil). İki taraf aynı stile inince embedding'ler
         karşılaştırılabilir olmalı. GPU'da çalışır, tek süreç.

Her iki mod da mevcut çıktıları atlar (yarıda kesilirse kaldığı yerden devam).

Kullanım:
  python scripts/06_make_variants.py thick --kernel 3
  python scripts/06_make_variants.py thick --kernel 5
  python scripts/06_make_variants.py hed
  (+ --limit N: ilk N dosyayla deneme)
"""
import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def thicken_one(src: str, dst: str, kernel: int):
    """Worker: koyu çizgileri kalınlaştırır + hafif blur."""
    try:
        with Image.open(src) as im:
            im.load()
            arr = np.asarray(im.convert("L"))
        # beyaz zeminde koyu çizgi: erode koyu bölgeyi büyütür
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        arr = cv2.erode(arr, k)
        arr = cv2.GaussianBlur(arr, (3, 3), 0.8)
        Image.fromarray(arr).convert("RGB").save(dst)
        return Path(src).name, None
    except Exception:
        return Path(src).name, traceback.format_exc(limit=1).strip().splitlines()[-1]


def run_thick(files, dst_dir: Path, kernel: int):
    errors = []
    with ProcessPoolExecutor() as pool:
        futs = [pool.submit(thicken_one, str(f), str(dst_dir / f.name), kernel) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"thick{kernel}", unit="img"):
            name, err = fut.result()
            if err:
                errors.append(f"{name}\t{err}")
    return errors


def run_hed(files, dst_dir: Path, target_size: int):
    """CAD görsellerini sorgu tarafıyla aynı HED + invert işleminden geçirir."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "photo_to_lineart", ROOT / "scripts" / "02_photo_to_lineart.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.get_hed()  # modeli baştan yükle

    errors = []
    for f in tqdm(files, desc="hed", unit="img"):
        try:
            with Image.open(f) as im:
                im.load()
                img = im.convert("RGB")
            # sorgu tarafındaki to_lineart ile birebir aynı işlem (invert dahil)
            out = mod.to_lineart(img, target_size)
            out.save(dst_dir / f.name)
        except Exception:
            errors.append(f"{f.name}\t" + traceback.format_exc(limit=1).strip().splitlines()[-1])
    return errors


def main():
    parser = argparse.ArgumentParser(description="CAD veri varyantları")
    parser.add_argument("mode", choices=["thick", "hed"])
    parser.add_argument("--kernel", type=int, default=3, help="thick modunda çekirdek boyutu")
    parser.add_argument("--limit", type=int, default=None, help="İlk N dosyayla deneme")
    parser.add_argument("--src", type=str, default=None,
                        help="Kaynak klasör (varsayılan cad_clean); ör. hed'i thick5 üstüne "
                             "uygulamak için: hed --src data/variants/cad_thick5 --name cad_hedthick5")
    parser.add_argument("--name", type=str, default=None, help="Çıktı varyant klasör adı")
    args = parser.parse_args()

    cfg = load_config()
    src_dir = (ROOT / args.src) if args.src else (ROOT / cfg["paths"]["cad_clean"])
    name = args.name or (f"cad_thick{args.kernel}" if args.mode == "thick" else "cad_hed")
    dst_dir = ROOT / "data" / "variants" / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.png"))
    if args.limit is not None:
        files = files[:args.limit]
    files = [f for f in files if not (dst_dir / f.name).exists()]  # devam mantığı
    if not files:
        print(f"{name}: tüm dosyalar zaten üretilmiş.")
        return
    print(f"{name}: {len(files)} dosya üretilecek -> {dst_dir}")

    if args.mode == "thick":
        errors = run_thick(files, dst_dir, args.kernel)
    else:
        errors = run_hed(files, dst_dir, cfg["image_size"])

    if errors:
        (dst_dir / "_errors.log").write_text("\n".join(errors), encoding="utf-8")
        print(f"{len(errors)} hata -> {dst_dir / '_errors.log'}")
    print(f"Bitti: {len(files) - len(errors)}/{len(files)}")


if __name__ == "__main__":
    main()
