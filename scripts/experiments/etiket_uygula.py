# -*- coding: utf-8 -*-
"""
etiket_uygula.py — etiketle.html'den indirilen etiket_onay.json'ı işler.

  - Seçilen çiftler labels_clean.json'a eklenir:
      öneriden seçilen -> "model_reviewed"   (eğitimde kullanılır, teste girmez)
      aramayla bulunan -> "search_reviewed"  (eğitimde kullanılır, teste girmez)
  - "Eşleşme yok" işaretlenen fotolar data/eval/etiketleme/atlananlar.json'a
    yazılır; sayfa yeniden kurulunca listeden düşerler.

Kullanım: python scripts/experiments/etiket_uygula.py <indirilen.json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from label_tools import ensure_kaynak, load_labels_base, save_labels_clean  # noqa: E402

OUT_DIR = ROOT / "data" / "eval" / "etiketleme"
TAG = {"oneri": "model_reviewed", "arama": "search_reviewed"}


def main():
    if len(sys.argv) != 2:
        sys.exit("Kullanım: python etiket_uygula.py <etiket_onay.json>")
    with open(sys.argv[1], encoding="utf-8-sig") as f:
        marks = json.load(f)
    onay, tur, yok = marks.get("onay", {}), marks.get("tur", {}), marks.get("yok", [])

    data = load_labels_base()
    eslesme = data.setdefault("eslesme", {})
    kaynak = ensure_kaynak(data)
    cad_dir = ROOT / "data" / "cad_clean"

    added, skipped = 0, 0
    for foto, cads in onay.items():
        if not (ROOT / "photos" / foto).exists():
            print(f"  ATLANDI: {foto} (photos/ altında yok)")
            skipped += len(cads)
            continue
        for cad in cads:
            if not (cad_dir / cad).exists():
                print(f"  ATLANDI: {foto} -> {cad} (cad_clean'de yok)")
                skipped += 1
                continue
            cur = eslesme.setdefault(foto, [])
            if cad not in cur:
                cur.append(cad)
                kaynak.setdefault(foto, {})[cad] = TAG.get(
                    tur.get(foto, {}).get(cad), "model_reviewed")
                added += 1
        if foto in eslesme:
            eslesme[foto] = sorted(eslesme[foto])

    atla_path = OUT_DIR / "atlananlar.json"
    eski = set(json.load(open(atla_path, encoding="utf-8"))) if atla_path.exists() else set()
    eski |= set(yok)
    eski -= set(eslesme)  # sonradan etiketlenen atlanmış sayılmaz
    with open(atla_path, "w", encoding="utf-8") as f:
        json.dump(sorted(eski), f, ensure_ascii=False, indent=1)

    print(f"\n{added} çift eklendi, {skipped} atlandı; "
          f"{len(yok)} foto 'eşleşme yok' listesinde (toplam {len(eski)}).")
    save_labels_clean(data)
    print("Sayfayı tazelemek için: python scripts/experiments/etiket_sayfa_kur.py")


if __name__ == "__main__":
    main()
