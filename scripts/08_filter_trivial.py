# -*- coding: utf-8 -*-
"""
08_filter_trivial.py — İndeksten parça profillerini (trivial CAD) ayıklama.

data/cad_clean'deki her görsel için basitlik metrikleri hesaplar:
  - ink      : koyu piksel oranı (çizgi yoğunluğu)
  - contours : anlamlı kontur sayısı (parça profilleri birkaç boş dikdörtgen)
  - elong    : ORİJİNAL cad_png'nin en-boy oranı max(w/h, h/w)
               (parça profilleri genelde aşırı yatay/dikey export edilir)

Bunlardan yüzdelik tabanlı bir "trivial skoru" üretir (1'e yakın = basit).
En basit N görsel data/debug/trivial_review.html'e döşenir; ✓ işaretlenenler
"İşaretleri indir" ile JSON alınıp şu komutla indeks dışına alınır:

  python scripts/08_filter_trivial.py --apply İndirilenler/trivial_disla.json

--apply, index/excluded_trivial.json'ı yazar (mevcutla BİRLEŞTİRİR). Dosya
silinmez; config'te exclude_trivial: true yapılınca arama ve yeni indeks
kurulumları bu listeyi dışlar. Öncesi/sonrası karşılaştırma:

  python scripts/04_evaluate.py --exclude-trivial off
  python scripts/04_evaluate.py --exclude-trivial on

Kullanım:
  python scripts/08_filter_trivial.py              # tara + onay sayfası + hub raporu
  python scripts/08_filter_trivial.py --top 300    # en basit 300 görseli döşe
  python scripts/08_filter_trivial.py --rescan     # metrik cache'ini yeniden hesapla
  python scripts/08_filter_trivial.py --bulk-name-pattern
        # kontur<=2 VE adında par/cizgi geçenleri excluded listesine toplu ekle
  python scripts/08_filter_trivial.py --remaining
        # kontur<=2 olup isim kalıbına uymayan ve henüz dışlanmamış kalanları
        # trivial_review_remaining.html'e döşe (işaretler ana sayfayla ortak)
  python scripts/08_filter_trivial.py --apply f.json [--replace]
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DARK_THRESHOLD = 200   # 01_clean_cad ile aynı içerik eşiği
MIN_CONTOUR_AREA = 20  # gürültü konturlarını sayma
METRICS_CACHE = ROOT / "data" / "eval" / "trivial_metrics.json"
EXCLUDED_PATH = ROOT / "index" / "excluded_trivial.json"
# skor ağırlıkları: az mürekkep ve az kontur "basit", aşırı en-boy "profil" işareti
W_INK, W_CONTOURS, W_ELONG = 0.4, 0.3, 0.3
# toplu dışlama kuralı: bu kontur eşiğinin altında VE adı bu kalıba uyan dosyalar
BULK_MAX_CONTOURS = 2
# 'par'/'cizgi' geçen her isim eşleşir (substring). Gözle doğrulandı (2026-07-04):
# 150 genel + 50 yüksek-konturlu + 60 near-miss örnekleminin TAMAMI parça profili
# çıktı; katalogda PARLAK/APART tipi masum isim yok, 'par' sonrası harfler hep
# alan-içi kod ekleri (pars, parkab, parsag/parsol, parters, pararka, parbos...).
# Kataloğa kelime içinde 'par' geçen GERÇEK model adı eklenirse burayı daralt
# (eski sınır kurallı hali: r"(par|cizgi|çizgi)(?![a-zçğıiöşü])").
BULK_NAME_RE = re.compile(r"par|cizgi|çizgi", re.IGNORECASE)


def name_pattern_match(filename: str) -> bool:
    stem = filename[:-4] if filename.lower().endswith(".png") else filename
    return bool(BULK_NAME_RE.search(stem))


def measure_one(clean_path: str, png_path: str):
    """Worker: (dosya adı, ink, contours, elong) veya hata döner."""
    name = Path(clean_path).name
    try:
        with Image.open(clean_path) as im:
            arr = np.asarray(im.convert("L"))
        mask = (arr < DARK_THRESHOLD).astype(np.uint8)
        ink = float(mask.mean())
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        n_cont = sum(1 for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA)
        elong = 1.0
        p = Path(png_path)
        if p.exists():
            with Image.open(p) as im:  # sadece header okunur, hızlı
                w, h = im.size
            if w > 0 and h > 0:
                elong = float(max(w / h, h / w))
        return name, ink, n_cont, elong, None
    except Exception as e:
        return name, 0.0, 0, 1.0, f"{type(e).__name__}: {e}"


def compute_metrics(rescan: bool) -> dict:
    """Tüm cad_clean için metrikler; cache varsa oradan okur."""
    if METRICS_CACHE.exists() and not rescan:
        with open(METRICS_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Metrik cache'i: {METRICS_CACHE.relative_to(ROOT)} ({len(data)} kayıt) "
              f"— yeniden hesap için --rescan")
        return data

    clean_dir = ROOT / "data" / "cad_clean"
    png_dir = ROOT / "cad_png"
    files = sorted(clean_dir.glob("*.png"))
    print(f"{len(files)} görsel taranıyor...")
    data, errors = {}, []
    with ProcessPoolExecutor() as pool:
        futs = [pool.submit(measure_one, str(f), str(png_dir / f.name)) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Basitlik", unit="img"):
            name, ink, n_cont, elong, err = fut.result()
            if err:
                errors.append(f"{name}\t{err}")
                continue
            data[name] = {"ink": round(ink, 5), "contours": n_cont,
                          "elong": round(elong, 3)}
    if errors:
        print(f"{len(errors)} dosya okunamadı (ilk 5): {errors[:5]}")

    add_scores(data)
    METRICS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Metrikler -> {METRICS_CACHE.relative_to(ROOT)}")
    return data


def add_scores(data: dict):
    """Yüzdelik tabanlı trivial skoru (1'e yakın = basit) her kayda eklenir."""
    names = list(data)
    ink = np.array([data[n]["ink"] for n in names])
    cont = np.array([data[n]["contours"] for n in names], dtype=float)
    elong = np.array([data[n]["elong"] for n in names])

    def pct(x):  # 0..1 arası sıra yüzdeliği (bağlar ortalama sırayla)
        order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        return order / max(len(x) - 1, 1)

    score = W_INK * (1 - pct(ink)) + W_CONTOURS * (1 - pct(cont)) + W_ELONG * pct(elong)
    for n, s in zip(names, score):
        data[n]["score"] = round(float(s), 4)


def hub_report(data: dict):
    """Son değerlendirmenin top-5'lerinde en sık geçen CAD'lerin skor/sıralarını yazar."""
    files = sorted((ROOT / "data" / "eval").glob("results_*.json"))
    if not files:
        print("Hub raporu: results_*.json yok, atlanıyor.")
        return
    with open(files[-1], "r", encoding="utf-8") as f:
        res = json.load(f)
    counts = Counter(t["cad"] for s in res["ornekler"] for t in s["top5"])

    ranked = sorted(data, key=lambda n: -data[n]["score"])  # 1. sıra = en basit
    rank_of = {n: i + 1 for i, n in enumerate(ranked)}
    print(f"\nHub raporu ({files[-1].name} top-5 sayımları) — trivial sırası / {len(ranked)}:")
    print(f"{'top5':>5}  {'skor':>6}  {'sıra':>6}  {'ink':>7}  {'kontur':>6}  {'en-boy':>6}  dosya")
    for name, c in counts.most_common(15):
        m = data.get(name)
        if m is None:
            print(f"{c:5d}  {'—':>6}  metrik yok: {name}")
            continue
        print(f"{c:5d}  {m['score']:6.3f}  {rank_of[name]:6d}  {m['ink']:7.4f}  "
              f"{m['contours']:6d}  {m['elong']:6.2f}  {name}")


_PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Trivial CAD onayı</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f2f2f2}
 header{position:sticky;top:0;background:#1e293b;color:#fff;padding:10px 16px;
        display:flex;gap:16px;align-items:center;z-index:5;flex-wrap:wrap}
 header b{color:#7dd3fc}
 button{cursor:pointer;border:0;border-radius:6px;padding:8px 14px;font-size:14px}
 #export{background:#059669;color:#fff} #all{background:#0369a1;color:#fff}
 #clear{background:#475569;color:#fff}
 .grid{display:flex;flex-wrap:wrap;gap:10px;padding:12px}
 .cell{width:200px;background:#fff;border:2px solid #ddd;border-radius:8px;
       padding:6px;text-align:center;cursor:pointer}
 .cell img{max-width:186px;max-height:240px;background:#fff}
 .name{font-size:11px;word-break:break-all;color:#444}
 .met{font-size:10px;color:#888}
 .cell.out{border-color:#dc2626;background:#fef2f2}
 .hint{font-size:12px;color:#cbd5e1}
</style></head><body>
<header>
 <span><b id="n-out">0</b> dışlanacak / <b>__TOTAL__</b></span>
 <button id="all">Tümünü işaretle</button>
 <button id="clear">Tümünü kaldır</button>
 <button id="export">İşaretleri indir (JSON)</button>
 <span class="hint">karta tıkla = dışla/geri al · en basitten karmaşığa sıralı · localStorage'da saklanır</span>
</header>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const KEY = "__KEY__";
const store = JSON.parse(localStorage.getItem(KEY) || "{}");
function save(){ localStorage.setItem(KEY, JSON.stringify(store)); refresh(); }
function refresh(){
  let n=0;
  document.querySelectorAll(".cell").forEach(c=>{
    const on = !!store[c.dataset.k];
    c.classList.toggle("out", on); if(on) n++;
  });
  document.getElementById("n-out").textContent = n;
}
const grid = document.getElementById("grid");
for(const d of DATA){
  const cell = document.createElement("div"); cell.className="cell"; cell.dataset.k=d.name;
  cell.innerHTML = `<img loading="lazy" src="${d.src}"><div class="name">${d.name}</div>
    <div class="met">skor ${d.score} · ink ${d.ink} · kontur ${d.contours} · oran ${d.elong}</div>`;
  cell.onclick = ()=>{ store[d.name] = store[d.name] ? undefined : 1; save(); };
  grid.appendChild(cell);
}
refresh();
document.getElementById("all").onclick  = ()=>{ for(const d of DATA) store[d.name]=1; save(); };
document.getElementById("clear").onclick= ()=>{ for(const k in store) delete store[k]; save(); };
document.getElementById("export").onclick = ()=>{
  const disla = Object.keys(store).filter(k=>store[k]);
  const blob = new Blob([JSON.stringify({disla},null,1)],{type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "__EXPORT__"; a.click();
};
</script></body></html>
"""


def load_excluded() -> set:
    if EXCLUDED_PATH.exists():
        with open(EXCLUDED_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_excluded(new: set, replace: bool, source: str):
    old = set() if replace else load_excluded()
    merged = sorted(old | new)
    EXCLUDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXCLUDED_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f"{source}: {len(new)} yeni + {len(old)} mevcut -> {len(merged)} kayıt "
          f"({EXCLUDED_PATH.relative_to(ROOT)})")


def bulk_selection(data: dict, max_contours: int = BULK_MAX_CONTOURS):
    """(kural ile dışlanacaklar, isim kalıbına uymayan kontur<=2 kalanlar).
    max_contours < 0 ise kontur şartı uygulanmaz (isim kalıbı tek başına yeter —
    kontrol: kontur>50 isimli dosyalar bile taralı profil/desen levhası çıktı)."""
    bulk = {n for n, m in data.items()
            if name_pattern_match(n)
            and (max_contours < 0 or m["contours"] <= max_contours)}
    rest = {n for n, m in data.items()
            if m["contours"] <= BULK_MAX_CONTOURS and not name_pattern_match(n)}
    return bulk, rest


def cmd_bulk_name_pattern(data: dict, replace: bool, max_contours: int):
    bulk, rest = bulk_selection(data, max_contours)
    kural = ("ad 'par/cizgi' içerir (kontur şartsız)" if max_contours < 0
             else f"kontur<={max_contours} VE ad 'par/cizgi' içerir")
    print(f"Kural ({kural}): {len(bulk)} dosya")
    save_excluded(bulk, replace, "bulk-name-pattern")
    already = load_excluded()
    remaining = rest - already
    print(f"İsim kalıbına uymayan kontur<={BULK_MAX_CONTOURS} kalan: {len(remaining)} "
          f"-> gözle bakmak için: python scripts/08_filter_trivial.py --remaining")


def build_review(data: dict, top_n, names=None, out_name="trivial_review.html",
                 store_key="trivial_review", export_name="trivial_disla.json",
                 shuffle_seed=None):
    """shuffle_seed verilirse skor sırası yerine rastgele sıra (örneklem sayfaları);
    store_key farklıysa işaretler ana trivial sayfasıyla KARIŞMAZ."""
    pool = names if names is not None else list(data)
    ranked = sorted(pool, key=lambda n: -data[n]["score"])
    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(ranked)
    if top_n:
        ranked = ranked[:top_n]
    out = ROOT / "data" / "debug" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ranked:
        m = data[n]
        rows.append({"name": n, "src": quote(f"../cad_clean/{n}"),
                     "score": m["score"], "ink": m["ink"],
                     "contours": m["contours"], "elong": m["elong"]})
    page = (_PAGE.replace("__TOTAL__", str(len(rows)))
            .replace("__KEY__", store_key)
            .replace("__EXPORT__", export_name)
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False)))
    out.write_text(page, encoding="utf-8")
    print(f"\nOnay sayfası: {out.relative_to(ROOT)} ({len(rows)} görsel)")


def cmd_pattern_sample(data: dict, n_sample: int, n_hicontour: int):
    """Toplu dışlama öncesi güvenlik kontrolü: kalıp tanımı, sayılar, iki örneklem
    sayfası (genel + yüksek konturlu riskli grup) ve daraltmanın dışarıda
    bıraktığı 'par+harf' near-miss grubunun dökümü."""
    print(f"Kalıp: r\"{BULK_NAME_RE.pattern}\" (IGNORECASE, uzantısız gövdede) —")
    print("  'par'/'cizgi' yalnız isim SONUNDA ya da rakam/ayraçtan önce eşleşir;")
    print("  ardından harf gelirse (PARLAK01, APART1, PARK05) eşleşmez.")

    matched = sorted(n for n in data if name_pattern_match(n))
    loose_re = re.compile(r"par|cizgi|çizgi", re.IGNORECASE)
    near_miss = sorted(n for n in data if loose_re.search(n) and not name_pattern_match(n))
    print(f"\nDaraltılmış kalıba uyan: {len(matched)} dosya")
    print(f"Substring uyup daraltmayla ELENEN: {len(near_miss)} dosya — örnekler: "
          + ", ".join(near_miss[:12]))

    hi = [n for n in matched if data[n]["contours"] > 10]
    print(f"Kalıba uyan VE kontur>10 (riskli grup): {len(hi)} dosya")

    build_review(data, n_sample, names=matched,
                 out_name="name_pattern_sample.html",
                 store_key="name_pattern_sample",
                 export_name="name_pattern_isaretler.json", shuffle_seed=42)
    build_review(data, n_hicontour, names=hi,
                 out_name="name_pattern_sample_hicontour.html",
                 store_key="name_pattern_hicontour",
                 export_name="name_pattern_hicontour_isaretler.json", shuffle_seed=43)
    # daraltmanın dışarıda bıraktığı 'par+harf' grubu (pars, parkab, parters...)
    # kalıbı genişletme kararı için ayrı örneklemde gösterilir
    build_review(data, min(60, len(near_miss)), names=near_miss,
                 out_name="name_pattern_sample_nearmiss.html",
                 store_key="name_pattern_nearmiss",
                 export_name="name_pattern_nearmiss_isaretler.json", shuffle_seed=44)
    print("Örneklemler rastgele sırada; karta tıklayıp işaretledikleriniz (yanlış "
          "pozitifler) JSON'la indirilebilir.")


def cmd_apply(apply_path: Path, replace: bool):
    with open(apply_path, "r", encoding="utf-8-sig") as f:
        marks = json.load(f)
    new = set(marks.get("disla", marks if isinstance(marks, list) else []))
    if not new:
        print("UYARI: dosyada dışlanacak kayıt yok.")
        return
    save_excluded(new, replace, "apply")
    print("Etkinleştirmek için configs/config.yaml'da exclude_trivial: true yapın.")
    print("Karşılaştırma: python scripts/04_evaluate.py --exclude-trivial off / on")


def main():
    parser = argparse.ArgumentParser(description="Parça profili (trivial CAD) ayıklama")
    parser.add_argument("--top", type=int, default=None,
                        help="Onay sayfasına döşenecek sayı (varsayılan: 200; "
                             "--remaining'de tümü)")
    parser.add_argument("--rescan", action="store_true", help="Metrik cache'ini yeniden hesapla")
    parser.add_argument("--bulk-name-pattern", action="store_true",
                        help=f"kontur<={BULK_MAX_CONTOURS} VE adında par/cizgi geçenleri "
                             "excluded listesine toplu ekle")
    parser.add_argument("--bulk-max-contours", type=int, default=BULK_MAX_CONTOURS,
                        help="Toplu kuralın kontur eşiği; -1 = kontur şartsız, "
                             "isim kalıbı tek başına yeter")
    parser.add_argument("--pattern-sample", action="store_true",
                        help="Toplu dışlama öncesi güvenlik kontrolü: kalıp tanımı, "
                             "sayılar ve örneklem sayfaları (150 genel + 50 kontur>10)")
    parser.add_argument("--sample-size", type=int, default=150,
                        help="--pattern-sample genel örneklem boyutu")
    parser.add_argument("--sample-hicontour", type=int, default=50,
                        help="--pattern-sample yüksek konturlu örneklem boyutu")
    parser.add_argument("--remaining", action="store_true",
                        help="İsim kalıbına uymayan kontur<=2 kalanları ayrı sayfaya döşe")
    parser.add_argument("--apply", type=str, default=None,
                        help="Onay JSON'unu index/excluded_trivial.json'a işle")
    parser.add_argument("--replace", action="store_true",
                        help="Birleştirme yerine üstüne yaz (--apply/--bulk-name-pattern)")
    args = parser.parse_args()

    if args.apply:
        cmd_apply(Path(args.apply), args.replace)
        return

    data = compute_metrics(args.rescan)
    if "score" not in next(iter(data.values())):
        add_scores(data)

    if args.pattern_sample:
        cmd_pattern_sample(data, args.sample_size, args.sample_hicontour)
        return
    if args.bulk_name_pattern:
        cmd_bulk_name_pattern(data, args.replace, args.bulk_max_contours)
        return
    if args.remaining:
        _bulk, rest = bulk_selection(data)
        remaining = sorted(rest - load_excluded())
        print(f"İsim kalıbına uymayan kontur<={BULK_MAX_CONTOURS} kalan: {len(remaining)}")
        # aynı localStorage anahtarı kullanılır: iki sayfanın işaretleri ortak
        # birikir, "İşaretleri indir" hepsini tek JSON'da verir
        build_review(data, args.top, names=remaining,
                     out_name="trivial_review_remaining.html")
        return

    build_review(data, args.top or 200)
    hub_report(data)


if __name__ == "__main__":
    main()
