# -*- coding: utf-8 -*-
"""
10_assist_label.py — Model destekli etiketleme (active learning turu).

Etiketsiz her fotoğraf için projected indeksin top-K tahminini onay sayfasına
döker; ✓'lenenler labels_clean.json'a "model_reviewed" kaynak etiketiyle eklenir.

ÖNEMLİ — model_reviewed etiketi BİLEREK "temiz" (CLEAN_TAGS) sayılmaz:
adaylar modelin kendi tahmini olduğundan modelin zaten bildiği kolay örneklere
yanlıdır; test kümesine girerlerse metrik yapay şişer. Bu etiketler yalnız
EĞİTİM verisi olarak kullanılır (07 test ailelerini manual/prefix_reviewed'dan
seçmeye devam eder).

Akış:
  python scripts/10_assist_label.py                # işle + aday üret + onay sayfası
  python scripts/10_assist_label.py --pages-only   # sadece sayfaları yeniden kur
  python scripts/10_assist_label.py --apply onay.json

Notlar:
  - Fotoğraf işleme sonuçları 04 ile AYNI cache'i kullanır (data/eval/cache);
    işlenmiş fotoğraf bir daha işlenmez. İlk koşu ~1.960 foto x ~9 sn sürer.
  - Adaylar data/eval/assist_candidates.json'a her 50 fotoğrafta bir yazılır;
    yarıda kesilirse kaldığı yerden devam eder.
  - Onay sayfasında fotoğraflar top-1 benzerlik skoruna göre (yüksekten alçağa)
    sıralanır: en güvenli öneriler başta — kalite düştüğünde bırakabilirsiniz.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from label_tools import (build_review_html, ensure_kaynak, load_labels_base,  # noqa: E402
                         save_labels_clean)

PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CANDIDATES_PATH = ROOT / "data" / "eval" / "assist_candidates.json"
PAGE_PATH = ROOT / "data" / "eval" / "assist_review.html"
TOP_K = 5
SAVE_EVERY = 50


def load_candidates() -> dict:
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_candidates(cands: dict):
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)


def cmd_generate(args):
    from search import get_engine
    data = load_labels_base()
    labeled = set(data.get("eslesme", {}))
    photos = sorted(p for p in (ROOT / "photos").iterdir()
                    if p.suffix.lower() in PHOTO_EXTS and p.name not in labeled)
    cands = load_candidates()
    todo = [p for p in photos if p.name not in cands]
    print(f"Etiketsiz: {len(photos)} | işlenmiş: {len(photos) - len(todo)} | "
          f"kalan: {len(todo)} | foto başına öneri: {args.top_k}")

    if todo and not args.pages_only:
        # varyant verilmez: config'deki aktif indeks (üretim modeli) kullanılır
        engine = get_engine()
        if args.group:
            import core as app_mod  # group_results (ikiz tekilleştirme)
        cache_dir = ROOT / "data" / "eval" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        errors, since = 0, 0
        for p in tqdm(todo, desc="Aday üretimi", unit="foto"):
            try:
                cp = cache_dir / (p.name + ".lineart.png")
                if cp.exists():
                    with Image.open(cp) as im:
                        im.load()
                        lineart = im.convert("RGB")
                else:
                    lineart, _, _ = engine.prepare_query(p)
                    lineart.save(cp)
                if args.group:
                    # k*3 ham sonuç -> ikiz tekilleştirme -> k FARKLI tasarım
                    raw = engine.search_prepared(lineart, k=min(args.top_k * 3, 150))
                    results = [(n, s) for n, s, _v in
                               app_mod.group_results(raw, args.top_k)]
                else:
                    results = engine.search_prepared(lineart, k=args.top_k)
                cands[p.name] = [[n, round(s, 4)] for n, s in results]
            except Exception as e:
                errors += 1
                cands[p.name] = {"hata": str(e)[:200]}
            since += 1
            if since >= SAVE_EVERY:
                save_candidates(cands)
                since = 0
        save_candidates(cands)
        if errors:
            print(f"{errors} fotoğraf işlenemedi (assist_candidates.json'da 'hata' kaydı)")

    build_pages(cands)


def build_pages(cands: dict):
    ok = {p: v for p, v in cands.items() if isinstance(v, list) and v}
    # en güvenli öneriler başa: top-1 skora göre azalan sırala
    ordered = sorted(ok.items(), key=lambda kv: -kv[1][0][1])
    cad_clean = ROOT / "data" / "cad_clean"
    rows = []
    for photo, results in ordered:
        rows.append({
            "photo": photo,
            "photo_path": ROOT / "photos" / photo,
            "cads": [{"name": n,
                      "path": (cad_clean / n) if (cad_clean / n).exists()
                              else ROOT / "cad_png" / n,
                      "badge": f"{s:.2f}"} for n, s in results],
        })
    # 20'li önerilerde sayfa başına satırı azalt ki sayfa hantallaşmasın
    per_cad = max((len(r["cads"]) for r in rows), default=5)
    build_review_html(rows, PAGE_PATH, "Model önerileri — etiket onayı",
                      "assist_onay.json",
                      hint="✓ = aynı model (etikete eklenecek); rozet = benzerlik "
                           "skoru; sayfalar güvenden düşüğe sıralı",
                      page_size=250 if per_cad <= 5 else 120)


def cmd_apply(apply_path: Path):
    with open(apply_path, "r", encoding="utf-8-sig") as f:
        marks = json.load(f)
    onay = marks.get("onay", marks)
    if not onay:
        print("UYARI: dosyada onaylanmış çift yok.")
        return
    data = load_labels_base()
    eslesme = data.setdefault("eslesme", {})
    kaynak = ensure_kaynak(data)
    cad_dir = ROOT / "cad_png"
    added = 0
    for photo, cads in onay.items():
        for cad in cads:
            if not (cad_dir / cad).exists():
                print(f"  ATLANDI: {photo} -> {cad} (cad_png'de yok)")
                continue
            cur = eslesme.setdefault(photo, [])
            if cad not in cur:
                cur.append(cad)
                kaynak.setdefault(photo, {})[cad] = "model_reviewed"
                added += 1
        if photo in eslesme:
            eslesme[photo] = sorted(eslesme[photo])
    print(f"{added} çift 'model_reviewed' etiketiyle eklendi "
          f"(eğitimde kullanılır, test kümesine GİRMEZ).")
    save_labels_clean(data)


def main():
    parser = argparse.ArgumentParser(description="Model destekli etiketleme")
    parser.add_argument("--apply", type=str, default=None)
    parser.add_argument("--pages-only", action="store_true",
                        help="İşleme yapmadan mevcut adaylardan sayfaları kur")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Foto başına öneri sayısı (r@20 %%51 vs r@5 %%36 — "
                             "20 ile daha çok eşleşme bulunur, tarama yükü artar)")
    parser.add_argument("--group", action="store_true",
                        help="Önerileri ikiz-tekilleştir: k*3 ham sonuçtan k FARKLI "
                             "tasarım (uygulamadaki gruplu kart mantığı)")
    args = parser.parse_args()
    if args.apply:
        cmd_apply(Path(args.apply))
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
