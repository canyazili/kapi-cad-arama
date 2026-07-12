# -*- coding: utf-8 -*-
"""
label_tools.py — Etiket araçlarının ortak yardımcıları.

09_name_match.py ve 06_label_review.py buradan import eder:
  - normalize_name / photo_root / cad_keys : dosya adı normalizasyonu
  - name_matches                           : kök + önek eşleşme kuralı
  - build_review_html                      : ✓/✗ işaretlemeli onay sayfası
  - load_labels_base / save_labels_clean   : labels_clean.json okuma/yazma
    (labels_clean varsa onun üstüne çalışılır; orijinal labels.json'a dokunulmaz)

07_train_projection.py de aile ayrımı için photo_root'u kullanır.
"""
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]

# Türkçe karakter sadeleştirme (iki büyük/küçük hali de kapsanır; çeviriden
# SONRA lower() çağrılır ki 'İ'.lower()'ın ürettiği birleşik nokta sorunu yaşanmasın)
_TR_MAP = str.maketrans({
    "ö": "o", "Ö": "o", "ü": "u", "Ü": "u", "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "İ": "i",
})


def normalize_name(s: str) -> str:
    """Küçük harf + Türkçe karakter sadeleştirme + boşluk temizliği."""
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_TR_MAP).lower()
    return s.replace(" ", "")


def photo_root(filename: str) -> str:
    """Fotoğraf adının kökü: uzantı at, normalize et, '-1'/'-2' gibi çekim
    numarası son eklerini ve '(2)' kopya eklerini kırp."""
    stem = Path(filename).stem
    n = normalize_name(stem)
    n = re.sub(r"\(\d+\)$", "", n)
    n = re.sub(r"[-_.]\d$", "", n)  # tek haneli çekim numarası (-1..-9)
    return n


def cad_keys(filename: str):
    """CAD adı için karşılaştırma anahtarları. 'fu1008-50' gibi dosyalarda
    'fu' öneki katalogda görünmez; hem ham kök hem fu'suz hali denenir."""
    n = normalize_name(Path(filename).stem)
    keys = [n]
    if n.startswith("fu") and len(n) > 2 and n[2].isdigit():
        keys.append(n[2:])
    return keys


def _boundary_ok(last: str, nxt: str) -> bool:
    """Önek eşleşmesinde sınır kontrolü: 'btozel6' -> 'btozel68' gibi sayı/harf
    ortasından bölünmeleri reddeder; ayraç veya sınıf değişimi kabul edilir."""
    if not nxt.isalnum():
        return True  # '-', '_' gibi ayraç
    if last.isdigit() and nxt.isdigit():
        return False
    if last.isalpha() and nxt.isalpha():
        return False
    return True  # harf->rakam veya rakam->harf geçişi ('1071' -> '1071delta')


def name_matches(photo_key: str, cad_key: str) -> bool:
    """Tam kök veya sınır kurallı önek eşleşmesi."""
    if cad_key == photo_key:
        return True
    if cad_key.startswith(photo_key):
        return _boundary_ok(photo_key[-1], cad_key[len(photo_key)])
    return False


# ---------------------------------------------------------------- labels I/O

# labels_clean.json'daki her çiftin kaynağı ("kaynak" anahtarı altında):
#   manual          — orijinal elle etiketlenmiş 172 kayıt
#   exact_auto      — isim kökü birebir eşleşmeden toplu onay (09 --bulk-approve-exact)
#   prefix_reviewed — önek adayı, HTML onay turunda gözle doğrulandı (09 --apply)
VALID_TAGS = ("manual", "exact_auto", "prefix_reviewed")


def ensure_kaynak(data: dict) -> dict:
    """'kaynak' sözlüğünü garanti eder: {foto: {cad: etiket}}. Etiketi olmayan
    mevcut çiftler 'manual' sayılır (orijinal kayıtlar elle etiketlenmişti);
    eslesme'den silinmiş çiftlerin artık etiketleri temizlenir."""
    kaynak = data.setdefault("kaynak", {})
    eslesme = data.get("eslesme", {})
    for photo, cads in eslesme.items():
        k = kaynak.setdefault(photo, {})
        for cad in cads:
            k.setdefault(cad, "manual")
    for photo in list(kaynak):
        cads = set(eslesme.get(photo, []))
        kaynak[photo] = {c: t for c, t in kaynak[photo].items() if c in cads}
        if not kaynak[photo]:
            del kaynak[photo]
    return kaynak


def load_labels_base(cfg=None) -> dict:
    """Üzerine çalışılacak etiket sözlüğünü döner: labels_clean.json varsa o,
    yoksa orijinal labels.json'ın kopyası. Tam JSON (tüm anahtarlar) döner."""
    clean = ROOT / "data" / "eval" / "labels_clean.json"
    for path in (clean, ROOT / "labels.json"):
        if path.exists():
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            print(f"Etiket tabanı: {path.relative_to(ROOT)} "
                  f"({len(data.get('eslesme', {}))} kayıt)")
            return data
    raise FileNotFoundError("Ne labels_clean.json ne labels.json bulundu.")


def save_labels_clean(data: dict):
    path = ROOT / "data" / "eval" / "labels_clean.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"Yazıldı: {path.relative_to(ROOT)} ({len(data.get('eslesme', {}))} kayıt)")
    return path


# ------------------------------------------------------------- onay sayfası

def _img_src(path: Path, html_dir: Path) -> str:
    """HTML dosyasından görsele göreli, URL-encode edilmiş src üretir."""
    try:
        rel = path.relative_to(html_dir)
        return quote(str(rel).replace("\\", "/"))
    except ValueError:
        pass
    # ortak kökten ../ ile çık
    rel = Path(*[".."] * len(html_dir.relative_to(ROOT).parts)) / path.relative_to(ROOT)
    return quote(str(rel).replace("\\", "/"))


_PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f2f2f2}
 header{position:sticky;top:0;background:#1e293b;color:#fff;padding:10px 16px;
        display:flex;gap:16px;align-items:center;z-index:5;flex-wrap:wrap}
 header .stat{font-size:14px} header b{color:#7dd3fc}
 button{cursor:pointer;border:0;border-radius:6px;padding:8px 14px;font-size:14px}
 #export{background:#059669;color:#fff} #clear{background:#475569;color:#fff}
 .row{background:#fff;margin:14px;border-radius:10px;padding:12px;display:flex;gap:14px}
 .photo{flex:0 0 300px;text-align:center}
 .photo img{max-width:300px;max-height:380px;border:1px solid #ccc;background:#fff}
 .cads{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
 .cell{width:190px;text-align:center;border:2px solid #ddd;border-radius:8px;padding:6px}
 .cell img{max-width:176px;max-height:230px;background:#fff}
 .name{font-size:11px;word-break:break-all;color:#444;min-height:26px}
 .badge{display:inline-block;font-size:10px;background:#0369a1;color:#fff;
        border-radius:4px;padding:1px 6px;margin-bottom:2px}
 .btns{display:flex;gap:6px;justify-content:center;margin-top:4px}
 .btns button{padding:4px 16px;font-size:15px;background:#e2e8f0}
 .cell.ok{border-color:#059669;background:#ecfdf5}
 .cell.ok .yes{background:#059669;color:#fff}
 .cell.bad{border-color:#dc2626;background:#fef2f2}
 .cell.bad .no{background:#dc2626;color:#fff}
 .ph-name{font-size:12px;color:#333;word-break:break-all;margin-top:4px}
 .hint{font-size:12px;color:#cbd5e1}
</style></head><body>
<header>
 <span class="stat"><b id="n-ok">0</b> ✓</span>
 <span class="stat"><b id="n-bad">0</b> ✗</span>
 <span class="stat"><b id="n-rest">0</b> işaretsiz</span>
 <button id="export">İşaretleri indir (JSON)</button>
 <button id="clear">Tümünü sıfırla</button>
 <span class="hint">__HINT__ · işaretler tarayıcıda (localStorage) saklanır, sayfayı kapatıp açabilirsiniz</span>
 <nav style="margin-left:auto">__NAV__</nav>
</header>
<div id="rows"></div>
<script>
const DATA = __DATA__;
const KEY  = "__STOREKEY__";
const store = JSON.parse(localStorage.getItem(KEY) || "{}");
function save(){ localStorage.setItem(KEY, JSON.stringify(store)); refresh(); }
function refresh(){
  let ok=0,bad=0,tot=0;
  document.querySelectorAll(".cell").forEach(c=>{
    tot++; const st = store[c.dataset.k];
    c.classList.toggle("ok", st==="ok"); c.classList.toggle("bad", st==="bad");
    if(st==="ok")ok++; else if(st==="bad")bad++;
  });
  document.getElementById("n-ok").textContent=ok;
  document.getElementById("n-bad").textContent=bad;
  document.getElementById("n-rest").textContent=tot-ok-bad;
}
const rowsEl = document.getElementById("rows");
for(const row of DATA){
  const div = document.createElement("div"); div.className="row";
  const ph = document.createElement("div"); ph.className="photo";
  ph.innerHTML = `<img loading="lazy" src="${row.photo_src}"><div class="ph-name">${row.photo}</div>`;
  div.appendChild(ph);
  const cads = document.createElement("div"); cads.className="cads";
  for(const c of row.cads){
    const k = row.photo+"\\t"+c.name;
    const cell = document.createElement("div"); cell.className="cell"; cell.dataset.k=k;
    cell.innerHTML = `${c.badge?`<div class="badge">${c.badge}</div>`:""}
      <img loading="lazy" src="${c.src}"><div class="name">${c.name}</div>
      <div class="btns"><button class="yes">✓</button><button class="no">✗</button></div>`;
    cell.querySelector(".yes").onclick = ()=>{ store[k] = store[k]==="ok" ? undefined : "ok"; save(); };
    cell.querySelector(".no").onclick  = ()=>{ store[k] = store[k]==="bad"? undefined : "bad"; save(); };
    cads.appendChild(cell);
  }
  div.appendChild(cads); rowsEl.appendChild(div);
}
refresh();
document.getElementById("clear").onclick = ()=>{
  if(confirm("Tüm işaretler silinsin mi?")){ for(const k in store) delete store[k]; save(); }
};
document.getElementById("export").onclick = ()=>{
  const onay={}, ret={};
  for(const [k,v] of Object.entries(store)){
    if(!v) continue;
    const [photo,cad] = k.split("\\t");
    const d = v==="ok" ? onay : ret;
    (d[photo] = d[photo]||[]).push(cad);
  }
  const blob = new Blob([JSON.stringify({onay,ret},null,1)],{type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "__EXPORTNAME__"; a.click();
};
</script></body></html>
"""


def build_review_html(rows, out_path: Path, title: str, export_name: str,
                      hint: str = "✓ = doğru eşleşme, ✗ = hatalı",
                      page_size: int = 250):
    """rows: [{photo, photo_path(Path), cads:[{name, path(Path)}]}].
    ✓/✗ işaretlemeli, JSON dışa aktarmalı statik onay sayfası yazar.

    page_size'dan çok satır varsa _p2, _p3... diye ek sayfalara bölünür;
    işaretler localStorage'da ORTAK tutulur, "İşaretleri indir" hangi sayfadan
    basılırsa basılsın TÜM sayfaların işaretlerini dışa aktarır."""
    html_dir = out_path.parent
    html_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for r in rows:
        data.append({
            "photo": r["photo"],
            "photo_src": _img_src(r["photo_path"], html_dir),
            "cads": [{"name": c["name"], "src": _img_src(c["path"], html_dir),
                      "badge": c.get("badge")}
                     for c in r["cads"]],
        })

    chunks = [data[i:i + page_size] for i in range(0, len(data), page_size)] or [[]]
    paths = [out_path if i == 0 else
             out_path.with_name(f"{out_path.stem}_p{i + 1}{out_path.suffix}")
             for i in range(len(chunks))]
    for i, (chunk, path) in enumerate(zip(chunks, paths)):
        nav = ""
        if len(chunks) > 1:
            links = [f'<b style="color:#7dd3fc">{j + 1}</b>' if j == i else
                     f'<a style="color:#fff" href="{p.name}">{j + 1}</a>'
                     for j, p in enumerate(paths)]
            nav = "sayfa: " + " ".join(links)
        page = (_PAGE
                .replace("__TITLE__", f"{title} ({i + 1}/{len(chunks)})")
                .replace("__HINT__", hint)
                .replace("__NAV__", nav)
                .replace("__STOREKEY__", "review:" + export_name)
                .replace("__EXPORTNAME__", export_name)
                .replace("__DATA__", json.dumps(chunk, ensure_ascii=False)))
        path.write_text(page, encoding="utf-8")
    n_pairs = sum(len(r["cads"]) for r in rows)
    extra = f", {len(chunks)} sayfa" if len(chunks) > 1 else ""
    print(f"Onay sayfası: {out_path.relative_to(ROOT)} "
          f"({len(rows)} fotoğraf, {n_pairs} çift{extra}) — tarayıcıda açın.")
