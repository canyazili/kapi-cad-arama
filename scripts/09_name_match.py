# -*- coding: utf-8 -*-
"""
09_name_match.py — İsim tabanlı otomatik etiket adayları.

Fotoğraf ve CAD dosya adları normalize edilir (küçük harf, Türkçe karakter
sadeleştirme, uzantı ve '-1/-2' çekim eklerinin atılması) ve adaylar İKİ SINIFA
ayrılır:
  exact  — normalize kök birebir aynı ('BTÖZEL68' -> 'btozel68.png');
           örneklem kontrolünden sonra --bulk-approve-exact ile toplu eklenir.
  prefix — sınır kurallı önek eşleşmesi ('1071' -> 'fu1071delta-240');
           HTML onay akışında tek tek gözle doğrulanır.

labels_clean.json'a eklenen her çift kaynak etiketi taşır ("kaynak" anahtarı):
manual (orijinal 172) / exact_auto / prefix_reviewed. Orijinal labels.json'a
asla dokunulmaz.

Akış:
  python scripts/09_name_match.py                     # sınıf sayıları + prefix onay sayfası
  python scripts/09_name_match.py --sample-exact 200  # exact'ten 200'lük kontrol sayfası
  python scripts/09_name_match.py --bulk-approve-exact # TÜM exact çiftleri labels_clean'e ekle
  python scripts/09_name_match.py --apply onay.json   # HTML'den indirilen ✓'leri ekle (prefix_reviewed)
"""
import argparse
import json
import random
import sys
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from label_tools import (build_review_html, cad_keys, ensure_kaynak,  # noqa: E402
                         load_labels_base, name_matches, photo_root,
                         save_labels_clean)

PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp")
MIN_ROOT_LEN = 3  # bundan kısa kökler ('s1' vb.) çöp eşleşme üretir, atlanır
CANDIDATES_PATH = ROOT / "data" / "eval" / "name_match_candidates.json"


def collect_candidates(photos, cad_files, labeled, include_labeled, max_cands):
    """Her fotoğraf için {cad_adı: 'exact'|'prefix'} sözlüğü. Sıralı anahtar
    listesi üzerinde bisect ile önek taraması yapılır (kaba tarama yerine)."""
    key_to_files = {}
    for f in cad_files:
        for k in cad_keys(f.name):
            key_to_files.setdefault(k, set()).add(f.name)
    keys = sorted(key_to_files)

    results = {}      # foto adı -> {cad adı: sınıf}
    n_overflow = 0
    for p in photos:
        already = set(labeled.get(p.name, []))
        if p.name in labeled and not include_labeled:
            continue
        root = photo_root(p.name)
        if len(root) < MIN_ROOT_LEN:
            continue
        cands = {}
        i = bisect_left(keys, root)
        while i < len(keys) and keys[i].startswith(root):
            if name_matches(root, keys[i]):
                cls = "exact" if keys[i] == root else "prefix"
                for name in key_to_files[keys[i]]:
                    # aynı dosya hem exact hem prefix anahtarla gelirse exact kazanır
                    if cands.get(name) != "exact":
                        cands[name] = cls
            i += 1
        for name in already:
            cands.pop(name, None)
        if not cands:
            continue
        if len(cands) > max_cands:  # exact her zaman kalır, prefix'ten kırpılır
            n_overflow += 1
            keep = sorted(cands, key=lambda n: (cands[n] != "exact", n))[:max_cands]
            cands = {n: cands[n] for n in keep}
        results[p.name] = dict(sorted(cands.items()))
    return results, n_overflow


def cad_display_path(name: str) -> Path:
    clean = ROOT / "data" / "cad_clean" / name
    return clean if clean.exists() else ROOT / "cad_png" / name


def make_rows(pairs_by_photo):
    return [{"photo": photo,
             "photo_path": ROOT / "photos" / photo,
             "cads": [{"name": c, "path": cad_display_path(c)} for c in cads]}
            for photo, cads in sorted(pairs_by_photo.items()) if cads]


def get_candidates(args):
    """Adayları hesaplar (ve denetim için JSON'a yazar)."""
    data = load_labels_base()
    labeled = data.get("eslesme", {})
    photos = sorted(p for p in (ROOT / "photos").iterdir()
                    if p.suffix.lower() in PHOTO_EXTS)
    cad_files = sorted((ROOT / "cad_png").glob("*.png"))
    results, n_overflow = collect_candidates(
        photos, cad_files, labeled, args.include_labeled, args.max_cands)

    n_exact = sum(1 for c in results.values() for cls in c.values() if cls == "exact")
    n_prefix = sum(1 for c in results.values() for cls in c.values() if cls == "prefix")
    n_unlabeled = sum(1 for p in photos if p.name not in labeled)
    print(f"\nFotoğraf: {len(photos)} toplam, {n_unlabeled} etiketsiz")
    print(f"Aday bulunan fotoğraf: {len(results)}")
    print(f"Aday çiftler: {n_exact} exact + {n_prefix} prefix = {n_exact + n_prefix}")
    if n_overflow:
        print(f"({n_overflow} fotoğrafta aday sayısı {args.max_cands} ile sınırlandı)")

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"Aday listesi: {CANDIDATES_PATH.relative_to(ROOT)}")
    return results


def cmd_generate(args):
    """TÜM adaylar (exact + prefix) tek onay akışında; exact'ler 'birebir isim'
    rozetiyle ve satır başında gösterilir (örneklemde exact'te de ~%17 hata
    çıktı — toplu onay yerine hepsi elle seçiliyor)."""
    results = get_candidates(args)
    rows = []
    for photo, cands in sorted(results.items()):
        ordered = sorted(cands, key=lambda c: (cands[c] != "exact", c))  # exact önce
        rows.append({
            "photo": photo,
            "photo_path": ROOT / "photos" / photo,
            "cads": [{"name": c, "path": cad_display_path(c),
                      "badge": "birebir isim" if cands[c] == "exact" else None}
                     for c in ordered],
        })
    out = ROOT / "data" / "eval" / "name_match_review.html"
    build_review_html(rows, out,
                      "İsim eşleşme adayları — onay", "name_match_onay.json",
                      hint="✓ = aynı model (etikete eklenecek), ✗ = alakasız; "
                           "mavi rozet = isim kökü birebir aynı")


def cmd_sample_exact(args):
    results = get_candidates(args)
    pairs = [(p, c) for p, cands in results.items()
             for c, cls in cands.items() if cls == "exact"]
    random.Random(42).shuffle(pairs)
    pairs = pairs[:args.sample_exact]
    by_photo = {}
    for p, c in pairs:
        by_photo.setdefault(p, []).append(c)
    out = ROOT / "data" / "eval" / "name_match_exact_sample.html"
    build_review_html(make_rows(by_photo), out,
                      f"Exact örneklem ({len(pairs)} çift) — hata oranı kontrolü",
                      "exact_sample_isaretler.json",
                      hint="✗ = yanlış eşleşme (hata oranına bakılıyor); toplu onaya "
                           "bu sayfa uygulanmaz")
    print(f"{len(pairs)} çiftlik exact örneklemi hazır.")


def cmd_bulk_approve_exact(args):
    results = get_candidates(args)
    data = load_labels_base()
    eslesme = data.setdefault("eslesme", {})
    kaynak = ensure_kaynak(data)
    added = 0
    for photo, cands in results.items():
        for cad, cls in cands.items():
            if cls != "exact":
                continue
            cur = eslesme.setdefault(photo, [])
            if cad not in cur:
                cur.append(cad)
                kaynak.setdefault(photo, {})[cad] = "exact_auto"
                added += 1
        if photo in eslesme:
            eslesme[photo] = sorted(eslesme[photo])
    print(f"{added} exact çift 'exact_auto' etiketiyle eklendi.")
    save_labels_clean(data)


def cmd_apply(apply_path: Path, tag: str):
    with open(apply_path, "r", encoding="utf-8-sig") as f:
        marks = json.load(f)
    onay = marks.get("onay", marks)  # düz {foto: [cad,...]} da kabul edilir
    if not onay:
        print("UYARI: dosyada onaylanmış çift yok ('onay' boş).")
        return

    data = load_labels_base()
    eslesme = data.setdefault("eslesme", {})
    kaynak = ensure_kaynak(data)
    cad_dir = ROOT / "cad_png"
    added, skipped = 0, []
    for photo, cads in onay.items():
        for cad in cads:
            if not (cad_dir / cad).exists():
                skipped.append(f"{photo} -> {cad} (cad_png'de yok)")
                continue
            cur = eslesme.setdefault(photo, [])
            if cad not in cur:
                cur.append(cad)
                kaynak.setdefault(photo, {})[cad] = tag
                added += 1
        if photo in eslesme:
            eslesme[photo] = sorted(eslesme[photo])

    print(f"{added} çift '{tag}' etiketiyle eklendi ({len(onay)} fotoğraf).")
    for s in skipped:
        print(f"  ATLANDI: {s}")
    save_labels_clean(data)


def main():
    parser = argparse.ArgumentParser(description="İsim tabanlı etiket adayları")
    parser.add_argument("--apply", type=str, default=None,
                        help="Onay JSON'unu labels_clean.json'a işle")
    parser.add_argument("--tag", type=str, default="prefix_reviewed",
                        help="--apply ile eklenen çiftlerin kaynak etiketi")
    parser.add_argument("--sample-exact", type=int, default=None,
                        help="Exact sınıfından rastgele N çiftlik kontrol sayfası üret")
    parser.add_argument("--bulk-approve-exact", action="store_true",
                        help="TÜM exact çiftleri labels_clean'e ekle (örneklem onayından sonra)")
    parser.add_argument("--include-labeled", action="store_true",
                        help="Etiketli fotoğraflara da (mevcut çiftler hariç) aday öner")
    parser.add_argument("--max-cands", type=int, default=40,
                        help="Foto başına en fazla aday (varsayılan 40)")
    args = parser.parse_args()

    if args.apply:
        cmd_apply(Path(args.apply), args.tag)
    elif args.bulk_approve_exact:
        cmd_bulk_approve_exact(args)
    elif args.sample_exact:
        cmd_sample_exact(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
