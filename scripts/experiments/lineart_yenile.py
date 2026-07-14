# -*- coding: utf-8 -*-
"""
lineart_yenile.py — CLAHE kurtarmasından etkilenebilecek etiketli fotoların
lineart önbelleğini (data/eval/cache) güncel boru hattıyla yeniler.

Etkilenenler:
  - denetim raporunda tam_kare / zayıf lineart bayraklıları (eski boru hattı
    kırpık halde 0.012 altına düşmüştü — yeni hat önce CLAHE dener)
  - önbellek ink < 0.02 olanlar (güvenlik payı)
  - yeni etiketlenen fotolar (denetim kapsamı dışıydı)
Yenilenen adlar data/train_cache/dinov3/photos_dinov3.npz'den DÜŞÜLÜR ki
dinov3 embedding'i bayat kalmasın (07 eksikleri kendisi embedler).

Kullanım: python scripts/experiments/lineart_yenile.py
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"c:/Users/canya/Desktop/kapı")
sys.path.insert(0, str(ROOT / "scripts"))
m = importlib.import_module("02_photo_to_lineart")

Image.MAX_IMAGE_PIXELS = None
CACHE_DIR = ROOT / "data" / "eval" / "cache"
NPZ = ROOT / "data" / "train_cache" / "dinov3" / "photos_dinov3.npz"
INK_ESIK = 0.02


def main():
    cfg = m.load_config()
    target = cfg["image_size"]
    labels = json.load(open(ROOT / "data/eval/labels_clean.json", encoding="utf-8"))
    etiketli = sorted(labels["eslesme"])

    # denetim bayrakları (eski 2823 foto için)
    rapor_p = ROOT / "data/debug/kirpma_denetim/rapor.json"
    audit = {}
    if rapor_p.exists():
        for r in json.load(open(rapor_p, encoding="utf-8"))["sonuc"]:
            audit[r["foto"]] = r
    yeni_fotolar = {p for p in etiketli if p not in audit}

    hedef = []
    for name in etiketli:
        cp = CACHE_DIR / (name + ".lineart.png")
        if not cp.exists():
            hedef.append((name, "cache_yok"))
            continue
        a = audit.get(name)
        if a and (a["tam_kare"] or a["ink"] < m.MIN_INK_RATIO):
            hedef.append((name, "denetim_bayragi"))
            continue
        with Image.open(cp) as im:
            im.load()
            ink = m.ink_ratio(im)
        if ink < INK_ESIK:
            hedef.append((name, f"ink_{ink:.4f}"))
        elif name in yeni_fotolar:
            # denetim görmemiş yeni foto: eski hat tam kareye düşmüş olabilir,
            # ucuz olduğundan yeniden üret
            hedef.append((name, "yeni_etiket"))

    print(f"Etiketli {len(etiketli)} foto; yenilenecek: {len(hedef)}", flush=True)
    from collections import Counter
    print(Counter(k.split('_')[0] for _, k in hedef), flush=True)

    m.get_hed()
    m.get_ocr()
    t0 = time.time()
    degisen, hata = [], []
    for i, (name, sebep) in enumerate(hedef):
        try:
            yeni = m.process_photo(ROOT / "photos" / name, target)
            cp = CACHE_DIR / (name + ".lineart.png")
            eski_ink = None
            if cp.exists():
                with Image.open(cp) as im:
                    im.load()
                    eski_ink = m.ink_ratio(im)
                    ayni = list(im.convert("L").getdata()) == list(yeni.convert("L").getdata())
            else:
                ayni = False
            if not ayni:
                yeni.save(cp)
                degisen.append(name)
                print(f"  degisti: {name} ink {eski_ink if eski_ink is None else round(eski_ink,4)} "
                      f"-> {m.ink_ratio(yeni):.4f} ({sebep})", flush=True)
        except Exception as e:
            hata.append(name)
            print(f"  HATA {name}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(hedef)}  {(time.time()-t0)/60:.1f} dk", flush=True)

    # dinov3 npz'den değişenleri düş
    if degisen and NPZ.exists():
        d = np.load(NPZ, allow_pickle=False)
        names = d["filenames"].tolist()
        keep = [i for i, n in enumerate(names) if n not in set(degisen)]
        np.savez(NPZ, embeddings=d["embeddings"][keep],
                 filenames=np.array([names[i] for i in keep]))
        print(f"photos_dinov3.npz: {len(names)} -> {len(keep)} "
              f"({len(names)-len(keep)} bayat kayıt düşüldü)", flush=True)

    with open(ROOT / "data/eval/etiketleme/lineart_yenileme.json", "w", encoding="utf-8") as f:
        json.dump({"degisen": degisen, "hata": hata,
                   "hedef": len(hedef)}, f, ensure_ascii=False, indent=1)
    print(f"BITTI: {len(hedef)} denendi, {len(degisen)} değişti, {len(hata)} hata, "
          f"{(time.time()-t0)/60:.1f} dk", flush=True)


if __name__ == "__main__":
    main()
