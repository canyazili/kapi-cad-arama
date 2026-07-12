# -*- coding: utf-8 -*-
"""
05_diagnose.py — Domain gap teşhis görselleri + sıralama analizleri.

1) Görseller (varsayılan): son results_*.json'dan 6 yanlış örnek yan yana
   (data/debug/diagnose.png) + top-5'lerde en sık geçen hub CAD şeridi
   (data/debug/hubs.png).

2) --ranks: her etiketli fotoğraf için doğru CAD'lerin EN İYİSİNİN aktif
   (füzyon) indeksteki TAM sırası — top-20'ye girmese de 50. mi 5000. mi
   görülür. Histogram + medyan yazdırılır, ayrıntı data/eval/rank_analysis.json.

3) --photo-photo: aynı CAD listesini paylaşan fotoğraf grupları (BAŞARI1020-1/-2
   gibi) embedding uzayında birbirini buluyor mu? Her fotoğrafın diğer etiketli
   fotoğraflar arasındaki top-5'inde grup arkadaşı var mı, oran nedir?

Kullanım:
  python scripts/05_diagnose.py                # sadece görseller (eski davranış)
  python scripts/05_diagnose.py --ranks --photo-photo
  python scripts/05_diagnose.py --all
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search import get_engine, load_config  # noqa: E402

PANEL = 300      # her panelin kenarı (px)
CAPTION_H = 34   # panel altı yazı alanı
N_SAMPLES = 6
N_HUBS = 5


def get_font(size=15):
    try:
        return ImageFont.truetype("arial.ttf", size)  # Türkçe karakterler için
    except OSError:
        return ImageFont.load_default()


def load_latest_results(eval_dir: Path) -> dict:
    files = sorted(eval_dir.glob("results_*.json"))
    if not files:
        print("UYARI: results_*.json yok. Önce scripts/04_evaluate.py çalıştırın.")
        sys.exit(0)
    with open(files[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Sonuç dosyası: {files[-1].name} ({data['ornek_sayisi']} örnek)")
    return data


def panel(img_path: Path, caption: str, font) -> Image.Image:
    """Tek panel: görsel + altında dosya adı."""
    canvas = Image.new("RGB", (PANEL, PANEL + CAPTION_H), (255, 255, 255))
    if img_path and img_path.exists():
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im.thumbnail((PANEL, PANEL))
            canvas.paste(im, ((PANEL - im.width) // 2, (PANEL - im.height) // 2))
    else:
        d = ImageDraw.Draw(canvas)
        d.text((10, PANEL // 2), "GÖRSEL YOK", fill=(200, 0, 0), font=font)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(180, 180, 180))
    d.text((4, PANEL + 4), caption[:44], fill=(0, 0, 0), font=font)
    return canvas


def load_labels() -> dict:
    """Etiketler: labels_clean.json varsa o, yoksa kökteki labels.json."""
    for path in (ROOT / "data" / "eval" / "labels_clean.json", ROOT / "labels.json"):
        if path.exists():
            with open(path, "r", encoding="utf-8-sig") as f:
                eslesme = json.load(f).get("eslesme", {})
            print(f"Etiketler: {path.name} ({len(eslesme)} kayıt)")
            return eslesme
    print("UYARI: etiket dosyası yok.")
    sys.exit(0)


def collect_query_embeddings(engine, photo_names, cache_dir: Path):
    """Etiketli fotoğrafların embedding'leri (cache'teki lineart'lardan; cache'te
    olmayan fotoğraf tam pipeline'dan geçirilir, ~9 sn/foto). name->vec döner."""
    photos_dir = ROOT / load_config()["paths"]["photos"]
    images, names, fresh = [], [], 0
    for name in photo_names:
        cache_path = cache_dir / (name + ".lineart.png")
        try:
            if cache_path.exists():
                with Image.open(cache_path) as im:
                    im.load()
                    images.append(im.convert("RGB"))
            else:
                lineart, _, _ = engine.prepare_query(photos_dir / name)
                lineart.save(cache_path)
                images.append(lineart)
                fresh += 1
            names.append(name)
        except Exception as e:
            print(f"  ATLANDI {name}: {e}")
    if fresh:
        print(f"({fresh} fotoğraf cache'te yoktu, yeniden işlendi)")
    embs = {}
    B = 32
    for i in range(0, len(images), B):
        batch = engine.embedder.embed_batch(images[i:i + B])
        for n, v in zip(names[i:i + B], batch):
            embs[n] = v
    return embs


def full_ranking(engine, q: np.ndarray):
    """Sorgu embedding'i için TÜM indeksin skor-sıralı dosya adı listesi."""
    if engine.index is None:  # füzyon
        scores = sum(w * (m @ q) for w, m in zip(engine.fusion_weights, engine.fusion_mats))
        order = np.argsort(-scores)
    else:
        _s, ids = engine.index.search(q.reshape(1, -1).astype(np.float32),
                                      engine.index.ntotal)
        order = ids[0]
    return [engine.filenames[i] for i in order]


def rank_analysis(engine, eslesme, embs, eval_dir: Path):
    """Her fotoğraf için doğru CAD'lerin en iyisinin tam sırası + histogram."""
    print("\n=== TAM SIRA ANALİZİ (doğru CAD'lerin en iyisi, tüm indeks) ===")
    known = set(engine.filenames)
    ranks, detail = [], {}
    for photo, cads in eslesme.items():
        if photo not in embs:
            continue
        targets = {Path(c).name for c in cads} & known
        if not targets:
            continue
        ordered = full_ranking(engine, embs[photo])
        rank = next((i for i, n in enumerate(ordered, 1) if n in targets), None)
        ranks.append(rank if rank is not None else len(ordered) + 1)
        detail[photo] = rank

    ranks = np.array(ranks)
    n_total = len(engine.filenames)
    bins = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100),
            (101, 500), (501, 2000), (2001, n_total)]
    print(f"{len(ranks)} fotoğraf | indeks boyutu {n_total}")
    print(f"medyan sıra: {int(np.median(ranks))} | ortalama: {ranks.mean():.0f}")
    for lo, hi in bins:
        c = int(((ranks >= lo) & (ranks <= hi)).sum())
        bar = "#" * round(60 * c / max(len(ranks), 1))
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        print(f"  {label:>10}: {c:4d} ({c / len(ranks):5.1%}) {bar}")

    out = eval_dir / "rank_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"varyant": engine.variant, "medyan": int(np.median(ranks)),
                   "siralar": detail}, f, ensure_ascii=False, indent=1)
    print(f"Ayrıntı -> {out}")


def photo_photo_check(eslesme, embs, eval_dir: Path):
    """Aynı CAD listesini paylaşan fotoğraflar birbirini top-5'te buluyor mu?"""
    print("\n=== FOTO-FOTO SAĞLAMASI (aynı CAD listesini paylaşanlar) ===")
    groups = defaultdict(list)
    for photo, cads in eslesme.items():
        if photo in embs:
            groups[frozenset(Path(c).name for c in cads)].append(photo)
    grouped = {p: gs for gs in groups.values() if len(gs) >= 2 for p in gs}
    print(f"{len(groups)} farklı CAD listesi; {sum(1 for g in groups.values() if len(g) >= 2)} "
          f"grupta >=2 fotoğraf ({len(grouped)} fotoğraf sağlamaya girer)")
    if not grouped:
        return

    names = [n for n in embs if n in grouped]
    Q = np.stack([embs[n] for n in names])
    sims = Q @ np.stack([embs[n] for n in embs]).T
    all_names = list(embs)
    self_idx = {n: all_names.index(n) for n in names}

    hits1 = hits5 = 0
    detail = []
    for i, n in enumerate(names):
        s = sims[i].copy()
        s[self_idx[n]] = -np.inf
        top = np.argsort(-s)[:5]
        top_names = [all_names[j] for j in top]
        mates = set(grouped[n]) - {n}
        h5 = any(t in mates for t in top_names)
        h1 = top_names[0] in mates
        hits5 += h5
        hits1 += h1
        detail.append({"photo": n, "grup": sorted(mates), "top5": top_names,
                       "top5_skor": [round(float(s[j]), 4) for j in top],
                       "top1_hit": h1, "top5_hit": h5})
    n_q = len(names)
    print(f"grup arkadaşı top-1'de: {hits1}/{n_q} ({hits1 / n_q:.1%})")
    print(f"grup arkadaşı top-5'te: {hits5}/{n_q} ({hits5 / n_q:.1%})")
    misses = [d for d in detail if not d["top5_hit"]][:5]
    if misses:
        print("bulamayan örnekler (ilk 5):")
        for d in misses:
            print(f"  {d['photo']} -> grup: {', '.join(d['grup'][:3])} | "
                  f"top-1: {d['top5'][0]}")
    out = eval_dir / "photo_photo.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"top1": hits1 / n_q, "top5": hits5 / n_q, "n": n_q,
                   "detay": detail}, f, ensure_ascii=False, indent=1)
    print(f"Ayrıntı -> {out}")


def make_visuals(cfg, eval_dir, cache_dir, cad_clean, debug_dir, font):
    data = load_latest_results(eval_dir)
    samples = data["ornekler"]

    # --- 1) 6 örnek: doğrusu 1. sırada olmayanlardan seç (yanlış tahmin görülebilsin)
    chosen = [s for s in samples if s["rank"] != 1][:N_SAMPLES]
    rows = []
    for s in chosen:
        lineart = cache_dir / (s["photo"] + ".lineart.png")
        target = cad_clean / s["targets"][0]
        top1 = cad_clean / s["top5"][0]["cad"]
        durum = "top-20 dışı" if s["rank"] is None else f"sıra {s['rank']}"
        row = [
            panel(lineart, f"SORGU: {s['photo']} ({durum})", font),
            panel(target, f"DOĞRU: {s['targets'][0]}", font),
            panel(top1, f"1.TAHMİN: {s['top5'][0]['cad']} ({s['top5'][0]['skor']:.2f})", font),
        ]
        rows.append(row)

    W = PANEL * 3 + 20 * 2
    H = (PANEL + CAPTION_H + 10) * len(rows)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    for r, row in enumerate(rows):
        for c, p in enumerate(row):
            sheet.paste(p, (c * (PANEL + 20), r * (PANEL + CAPTION_H + 10)))
    out1 = debug_dir / "diagnose.png"
    sheet.save(out1)
    print(f"Teşhis görseli -> {out1}")

    # --- 2) Hub analizi: top-5'lerde en sık geçen CAD'ler
    counts = Counter()
    for s in samples:
        for t in s["top5"]:
            counts[t["cad"]] += 1
    hubs = counts.most_common(N_HUBS)
    print("En sık top-5'e giren CAD'ler:")
    for name, c in hubs:
        print(f"  {c:3d}x  {name}")

    strip = Image.new("RGB", ((PANEL + 20) * N_HUBS, PANEL + CAPTION_H), (255, 255, 255))
    for i, (name, c) in enumerate(hubs):
        strip.paste(panel(cad_clean / name, f"{c}x  {name}", font), (i * (PANEL + 20), 0))
    out2 = debug_dir / "hubs.png"
    strip.save(out2)
    print(f"Hub görseli -> {out2}")


def main():
    parser = argparse.ArgumentParser(description="Teşhis görselleri ve sıralama analizleri")
    parser.add_argument("--ranks", action="store_true", help="Tam sıra analizi")
    parser.add_argument("--photo-photo", action="store_true", help="Foto-foto sağlaması")
    parser.add_argument("--all", action="store_true", help="Görseller + tüm analizler")
    parser.add_argument("--skip-visuals", action="store_true",
                        help="diagnose.png / hubs.png üretme")
    args = parser.parse_args()
    if args.all:
        args.ranks = args.photo_photo = True

    cfg = load_config()
    eval_dir = ROOT / cfg["paths"]["eval_dir"]
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cad_clean = ROOT / cfg["paths"]["cad_clean"]
    debug_dir = ROOT / cfg["paths"]["debug"]
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_visuals:
        make_visuals(cfg, eval_dir, cache_dir, cad_clean, debug_dir, get_font())

    if args.ranks or args.photo_photo:
        eslesme = load_labels()
        engine = get_engine()
        print(f"Varyant: {engine.variant} | model: {engine.embedder.model_name}")
        embs = collect_query_embeddings(engine, list(eslesme), cache_dir)
        if args.ranks:
            rank_analysis(engine, eslesme, embs, eval_dir)
        if args.photo_photo:
            photo_photo_check(eslesme, embs, eval_dir)


if __name__ == "__main__":
    main()
