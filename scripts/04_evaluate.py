# -*- coding: utf-8 -*-
"""
04_evaluate.py — labels.json tabanlı recall@k değerlendirmesi.

data/eval/labels.json (yoksa proje kökündeki labels.json) okunur:
  - "eslesme": { "foto_adı": ["doğru_cad_1.png", ...] }
    Her fotoğrafın BİRDEN FAZLA doğru CAD karşılığı olabilir (ölçü varyantları);
    top-k içinde bunlardan HERHANGİ BİRİ varsa isabet sayılır.

Fotoğraflar ./photos, CAD dosyaları ./cad_png altında aranır; karşılaştırma
sadece dosya adı (basename) üzerinden yapılır (cad_clean/cad_png aynı adları taşır).

Çıktılar (data/eval/):
  - missing.log        : bulunamayan fotoğraf/CAD kayıtları (her çalıştırmada yenilenir)
  - worst_cases.txt    : en kötü 10 örnek + top-5 tahminleri
  - results_<tarih>.json: metrikler + örnek bazlı sonuçlar (denemeleri karşılaştırmak için)
  - cache/             : işlenmiş sorgu lineart'ları (fotoğraf başına ~9 sn'lik
                         crop+OCR+HED adımı sadece ilk çalıştırmada yapılır)

Kullanım:
  python scripts/04_evaluate.py
  python scripts/04_evaluate.py --limit 20     # ilk 20 kayıtla hızlı deneme
  python scripts/04_evaluate.py --no-cache     # cache'i yok say (foto pipeline'ı değiştiyse!)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search import get_engine, load_config  # noqa: E402

K_VALUES = (1, 5, 10, 20)
WORST_N = 10


def load_labels(cfg):
    """Etiket dosyasını bulur ve 'eslesme' sözlüğünü döner.
    Öncelik: labels_clean.json (06/09 araçlarının temizlenmiş çıktısı) > config > kök."""
    candidates = [ROOT / "data" / "eval" / "labels_clean.json",
                  ROOT / cfg["paths"]["eval_labels"], ROOT / "labels.json"]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            eslesme = data.get("eslesme")
            if not eslesme:
                print(f"UYARI: {path} içinde 'eslesme' anahtarı yok ya da boş.")
                sys.exit(0)
            print(f"Etiket dosyası: {path} ({len(eslesme)} kayıt)")
            return eslesme
    print("UYARI: labels.json bulunamadı. Aranan yerler:\n  " +
          "\n  ".join(str(c) for c in candidates))
    sys.exit(0)


def get_query_lineart(engine, photo_path: Path, cache_dir: Path, use_cache: bool):
    """Fotoğrafın lineart halini üretir; varsa cache'ten okur, yoksa üretip yazar."""
    # Stem çakışmasın diye (1071.png vs 1071.jpeg) cache adı tam dosya adını taşır
    cache_path = cache_dir / (photo_path.name + ".lineart.png")
    if use_cache and cache_path.exists():
        with Image.open(cache_path) as im:
            im.load()
            return im.convert("RGB")
    lineart, _, _ = engine.prepare_query(photo_path)
    lineart.save(cache_path)
    return lineart


def main():
    parser = argparse.ArgumentParser(description="labels.json ile recall@k değerlendirmesi")
    parser.add_argument("--limit", type=int, default=None, help="İlk N kayıtla hızlı deneme")
    parser.add_argument("--no-cache", action="store_true",
                        help="Lineart cache'ini yok say, sorguları yeniden işle "
                             "(02'deki pipeline değiştiyse bunu kullanın)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Ölçülecek indeks varyantı (varsayılan: config'teki active_index)")
    parser.add_argument("--exclude-trivial", choices=["on", "off"], default=None,
                        help="Parça profili dışlamasını zorla aç/kapa "
                             "(varsayılan: config'teki exclude_trivial)")
    parser.add_argument("--photos-file", type=str, default=None,
                        help="Sadece bu JSON listedeki fotoğrafları ölç "
                             "(07'nin eğitim/test ayrımı için; düz liste ya da "
                             "{'photos': [...]} kabul edilir)")
    args = parser.parse_args()

    cfg = load_config()
    photos_dir = ROOT / cfg["paths"]["photos"]
    cad_dir = ROOT / cfg["paths"]["cad_png"]
    eval_dir = ROOT / cfg["paths"]["eval_dir"]
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    eslesme = load_labels(cfg)
    items = list(eslesme.items())
    if args.photos_file:
        with open(args.photos_file, "r", encoding="utf-8-sig") as f:
            keep = json.load(f)
        keep = set(keep["photos"] if isinstance(keep, dict) else keep)
        items = [(p, c) for p, c in items if p in keep]
        print(f"--photos-file: {len(items)}/{len(eslesme)} kayıt ölçülecek")
    if args.limit is not None:
        items = items[:args.limit]

    # Not: cache'te LINEART görselleri saklanır (embedding değil); lineart modelden
    # bağımsızdır, embedding her koşuda varyantın kendi modeliyle yeniden hesaplanır.
    # Bu yüzden model değişse de (vitb14 -> vitl14) cache güvenle geçerli kalır.
    excl = None if args.exclude_trivial is None else (args.exclude_trivial == "on")
    engine = get_engine(variant=args.variant, exclude_trivial=excl)
    print(f"Varyant: {engine.variant} | model: {engine.embedder.model_name} | "
          f"dışlanan trivial: {len(engine.excluded)}")
    max_k = max(K_VALUES)

    missing_lines = []   # bulunamayan dosyalar (missing.log)
    samples = []         # örnek bazlı sonuçlar
    hits = {k: 0 for k in K_VALUES}

    for photo_name, cad_list in tqdm(items, desc="Değerlendirme", unit="foto"):
        photo_path = photos_dir / photo_name
        if not photo_path.exists():
            missing_lines.append(f"FOTO YOK\t{photo_name}")
            continue

        targets, missing_cads = set(), []
        for cad in cad_list:
            name = Path(cad).name
            if (cad_dir / name).exists():
                targets.add(name)
            else:
                missing_cads.append(name)
        for name in missing_cads:
            missing_lines.append(f"CAD YOK\t{photo_name}\t{name}")
        if not targets:
            missing_lines.append(f"HIC GECERLI CAD KALMADI\t{photo_name}")
            continue

        try:
            lineart = get_query_lineart(engine, photo_path, cache_dir, not args.no_cache)
        except Exception as e:
            missing_lines.append(f"ISLEME HATASI\t{photo_name}\t{e}")
            continue

        results = engine.search_prepared(lineart, k=max_k)
        rank = None  # doğru CAD'lerden herhangi birinin ilk görüldüğü sıra
        for i, (name, _score) in enumerate(results, start=1):
            if Path(name).name in targets:
                rank = i
                break
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
        samples.append({
            "photo": photo_name,
            "rank": rank,
            "targets": sorted(targets),
            "top5": [{"cad": n, "skor": round(s, 4)} for n, s in results[:5]],
        })

    # --- missing.log (her çalıştırmada yenilenir; metrik bu kayıtlar OLMADAN hesaplanır)
    missing_log = eval_dir / "missing.log"
    if missing_lines:
        missing_log.write_text("\n".join(missing_lines) + "\n", encoding="utf-8")
        print(f"{len(missing_lines)} eksik/hatalı kayıt atlandı -> {missing_log}")
    elif missing_log.exists():
        missing_log.unlink()  # eski logu bırakma, bu koşuda eksik yok

    n = len(samples)
    if n == 0:
        print("UYARI: değerlendirilebilir örnek kalmadı.")
        sys.exit(0)

    # --- metrikler
    recalls = {f"recall@{k}": hits[k] / n for k in K_VALUES}
    print(f"\n{n} örnek değerlendirildi:")
    for k in K_VALUES:
        print(f"  recall@{k:<3}: {hits[k] / n:6.1%}  ({hits[k]}/{n})")

    # --- en kötü örnekler: hiç bulamayanlar önce, sonra en geç bulanlar
    worst = sorted(samples, key=lambda s: (s["rank"] is not None, -(s["rank"] or 0)))[:WORST_N]
    lines = [f"En kötü {len(worst)} örnek ({datetime.now():%Y-%m-%d %H:%M})", "=" * 60]
    for s in worst:
        durum = "BULUNAMADI (top-20 dışı)" if s["rank"] is None else f"sıra {s['rank']}"
        lines.append(f"\n{s['photo']}  ->  {durum}")
        lines.append(f"  doğrular : {', '.join(s['targets'])}")
        lines.append("  top-5    :")
        for t in s["top5"]:
            lines.append(f"    {t['skor']:.4f}  {t['cad']}")
    worst_path = eval_dir / "worst_cases.txt"
    worst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nEn kötü örnekler -> {worst_path}")

    # --- sonuç dosyası (denemeleri karşılaştırmak için)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_path = eval_dir / f"results_{stamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "tarih": stamp,
            "varyant": engine.variant,
            "ornek_sayisi": n,
            "atlanan_kayit": len(missing_lines),
            "model": engine.embedder.model_name,
            "exclude_trivial": len(engine.excluded),
            "photos_file": args.photos_file,
            "remove_text": cfg["text_removal"]["remove_text"],
            "recall": recalls,
            "ornekler": samples,
        }, f, ensure_ascii=False, indent=2)
    print(f"Sonuçlar -> {results_path}")


if __name__ == "__main__":
    main()
