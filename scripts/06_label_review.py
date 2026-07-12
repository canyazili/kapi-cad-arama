# -*- coding: utf-8 -*-
"""
06_label_review.py — Mevcut etiketleri gözle temizleme aracı.

labels.json'daki (labels_clean.json varsa onun üstünden) her foto->CAD eşleşmesini
foto+CAD yan yana gösteren bir onay sayfası üretir: data/eval/review.html.
Hatalı / parça profili olan çiftleri ✗ ile işaretleyip "İşaretleri indir" ile
JSON alın; sonra:

  python scripts/06_label_review.py --apply İndirilenler/review_isaretler.json

✗ işaretli çiftler çıkarılarak data/eval/labels_clean.json güncellenir
(orijinal labels.json'a DOKUNULMAZ; ✓ işaretleri sadece "kontrol edildi" notudur,
uygulamada bir şey değiştirmez).

Kullanım:
  python scripts/06_label_review.py                 # onay sayfası üret
  python scripts/06_label_review.py --apply isaretler.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from label_tools import (build_review_html, ensure_kaynak, load_labels_base,  # noqa: E402
                         save_labels_clean)


def cmd_generate():
    data = load_labels_base()
    eslesme = data.get("eslesme", {})
    cad_clean = ROOT / "data" / "cad_clean"
    cad_png = ROOT / "cad_png"

    rows = []
    for photo, cads in sorted(eslesme.items()):
        rows.append({
            "photo": photo,
            "photo_path": ROOT / "photos" / photo,
            "cads": [{"name": c,
                      "path": (cad_clean / c) if (cad_clean / c).exists() else cad_png / c}
                     for c in cads],
        })
    out = ROOT / "data" / "eval" / "review.html"
    build_review_html(rows, out, "Mevcut etiketler — temizlik",
                      "review_isaretler.json",
                      hint="✗ = hatalı / parça profili (çıkarılacak), ✓ = kontrol edildi")


def cmd_apply(apply_path: Path):
    with open(apply_path, "r", encoding="utf-8-sig") as f:
        marks = json.load(f)
    # onay sayfası {"onay":..., "ret":...} üretir; "hatali" veya düz sözlük de kabul
    bad = marks.get("ret") or marks.get("hatali") or {}
    if not bad and ("onay" not in marks and "ret" not in marks):
        bad = marks
    if not bad:
        print("UYARI: dosyada ✗ işaretli çift yok, değişiklik yapılmadı.")
        return

    data = load_labels_base()
    eslesme = data.get("eslesme", {})
    removed, emptied, missing = 0, [], []
    for photo, cads in bad.items():
        if photo not in eslesme:
            missing.append(photo)
            continue
        before = len(eslesme[photo])
        eslesme[photo] = [c for c in eslesme[photo] if c not in set(cads)]
        removed += before - len(eslesme[photo])
        if not eslesme[photo]:
            del eslesme[photo]
            emptied.append(photo)

    print(f"{removed} hatalı çift çıkarıldı.")
    if emptied:
        print(f"{len(emptied)} fotoğrafın hiç etiketi kalmadı, kayıt silindi: "
              + ", ".join(emptied[:10]) + ("..." if len(emptied) > 10 else ""))
    for m in missing:
        print(f"  UYARI: '{m}' etiketlerde yok, atlandı.")
    ensure_kaynak(data)  # silinen çiftlerin kaynak etiketlerini de temizler
    save_labels_clean(data)


def main():
    parser = argparse.ArgumentParser(description="Etiket temizleme onay aracı")
    parser.add_argument("--apply", type=str, default=None,
                        help="İşaret JSON'unu labels_clean.json'a uygula")
    args = parser.parse_args()
    if args.apply:
        cmd_apply(Path(args.apply))
    else:
        cmd_generate()


if __name__ == "__main__":
    main()
