# -*- coding: utf-8 -*-
"""
galeri_sayfa_kur.py — Tüm fotoğrafların işlenmiş SON HALİ galerisini kurar.

Girdi : data/debug/galeri/meta.jsonl + panel/*.jpg (galeri_uret.py çıktısı)
Çıktı : data/debug/galeri/galeri.html (+ _p2, _p3, ...)

Her kart: çizgi çıkarımı (aramada kullanılan hal) + foto adı + bayrak rozetleri.
Karta tıklayınca ORİJİNAL foto açılır (karşılaştırma için). "⚑" ile sorunlu
işaretlenir; "İşaretleri indir" tüm sayfaların işaretlerini tek JSON yapar.

Kullanım: python scripts/experiments/galeri_sayfa_kur.py [--sayfa-boyu 150]
"""
import argparse
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(r"c:/Users/canya/Desktop/kapı")
OUT_DIR = ROOT / "data" / "debug" / "galeri"

FLAG_TR = {
    "fallback_bant": ("kırpma bulunamadı", "#b45309"),
    "kontrast_kurtarma": ("kontrast kurtarması", "#0369a1"),
    "tam_kare": ("tam kare", "#7c3aed"),
    "zayif_lineart": ("çizgi çok az", "#dc2626"),
}

_PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kapı İşleme Galerisi __SAYFA__</title>
<style>
 *{box-sizing:border-box}
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#eef2f7}
 header{position:sticky;top:0;z-index:9;background:#1f3a5f;color:#fff;
        padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 header b{color:#7dd3fc}
 button{cursor:pointer;border:0;border-radius:6px;padding:8px 14px;font-size:14px}
 #export{background:#059669;color:#fff}
 #grid{display:flex;flex-wrap:wrap;gap:12px;padding:14px}
 .kart{width:200px;background:#fff;border:1px solid #d8dfea;border-radius:10px;
       padding:8px;text-align:center}
 .kart.flag{border:2px solid #c0392b;background:#fef2f2}
 .kart img{width:100%;height:184px;object-fit:contain;background:#fff;cursor:zoom-in}
 .ad{font-size:11px;word-break:break-all;color:#333;min-height:28px;margin:4px 0}
 .rozet{display:inline-block;font-size:10px;color:#fff;border-radius:4px;
        padding:1px 6px;margin:1px}
 .fbtn{background:#e2e8f0;padding:4px 10px;font-size:12px;width:100%}
 .kart.flag .fbtn{background:#c0392b;color:#fff}
 nav{margin-left:auto;font-size:13px;max-width:60%;line-height:1.8}
 nav a{color:#fff} nav b{color:#7dd3fc}
 .hint{font-size:12px;color:#cbd5e1}
 #onizleme{position:fixed;inset:0;background:#000c;display:none;z-index:99;
           align-items:center;justify-content:center;cursor:zoom-out;gap:10px}
 #onizleme img{max-width:47vw;max-height:94vh;background:#fff}
</style></head><body>
<header>
 <span><b id="n-flag">0</b> sorunlu işaretli</span>
 <button id="export">İşaretleri indir (JSON)</button>
 <span class="hint">kart = aramada kullanılan son hal · tıkla = orijinal fotoyla yan yana büyüt · ⚑ = sorunlu işaretle</span>
 <nav>__NAV__</nav>
</header>
<div id="grid"></div>
<div id="onizleme"><img id="on-foto"><img id="on-cizgi"></div>
<script>
const DATA = __DATA__;
const KEY = "galeri:v1";
const store = JSON.parse(localStorage.getItem(KEY) || "{}");
function save(){ localStorage.setItem(KEY, JSON.stringify(store)); refresh(); }
function refresh(){
  document.querySelectorAll(".kart").forEach(k=>{
    k.classList.toggle("flag", !!store[k.dataset.f]);
    k.querySelector(".fbtn").textContent = store[k.dataset.f] ? "⚑ Sorunlu (geri al)" : "⚑ Sorunlu";
  });
  document.getElementById("n-flag").textContent = Object.keys(store).filter(k=>store[k]).length;
}
const grid = document.getElementById("grid");
for(const r of DATA){
  const k = document.createElement("div"); k.className="kart"; k.dataset.f = r.foto;
  const roz = (r.rozet||[]).map(([t,c])=>`<span class="rozet" style="background:${c}">${t}</span>`).join("");
  k.innerHTML = `<img loading="lazy" src="${r.src}"><div class="ad">${r.foto}</div>
    <div>${roz}</div><button class="fbtn">⚑ Sorunlu</button>`;
  k.querySelector("img").onclick = ()=>{
    document.getElementById("on-foto").src = "../../../photos/" + encodeURIComponent(r.foto);
    document.getElementById("on-cizgi").src = r.src;
    document.getElementById("onizleme").style.display = "flex";
  };
  k.querySelector(".fbtn").onclick = ()=>{ store[r.foto] = !store[r.foto]; save(); };
  grid.appendChild(k);
}
refresh();
document.getElementById("onizleme").onclick = ()=>{ document.getElementById("onizleme").style.display="none"; };
document.addEventListener("keydown", e=>{ if(e.key==="Escape") document.getElementById("onizleme").style.display="none"; });
document.getElementById("export").onclick = ()=>{
  const list = Object.keys(store).filter(k=>store[k]).sort();
  const blob = new Blob([JSON.stringify({sorunlu:list},null,1)],{type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "galeri_sorunlu.json"; a.click();
};
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sayfa-boyu", type=int, default=150)
    args = ap.parse_args()

    rows, hatalar = [], []
    with open(OUT_DIR / "meta.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if "hata" in rec:
                hatalar.append(rec)
                continue
            panel = OUT_DIR / "panel" / (rec["foto"] + ".jpg")
            if not panel.exists():
                continue
            rows.append({
                "foto": rec["foto"],
                "src": "panel/" + quote(panel.name),
                "rozet": [FLAG_TR[b] for b in rec.get("bayrak", []) if b in FLAG_TR],
            })
    rows.sort(key=lambda r: r["foto"].lower())

    chunks = [rows[i:i + args.sayfa_boyu] for i in range(0, len(rows), args.sayfa_boyu)] or [[]]
    paths = [OUT_DIR / ("galeri.html" if i == 0 else f"galeri_p{i+1}.html")
             for i in range(len(chunks))]
    for i, (chunk, path) in enumerate(zip(chunks, paths)):
        links = []
        for j, p in enumerate(paths):
            links.append(f"<b>{j+1}</b>" if j == i else f'<a href="{p.name}">{j+1}</a>')
        nav = "sayfa: " + " ".join(links)
        page = (_PAGE
                .replace("__SAYFA__", f"({i+1}/{len(chunks)})")
                .replace("__NAV__", nav)
                .replace("__DATA__", json.dumps(chunk, ensure_ascii=False)))
        path.write_text(page, encoding="utf-8")

    print(f"Galeri hazır: {paths[0]}")
    print(f"  {len(rows)} foto, {len(chunks)} sayfa, {len(hatalar)} hata")
    if hatalar:
        for h in hatalar[:10]:
            print("  HATA:", h["foto"], "-", h.get("hata", ""))


if __name__ == "__main__":
    main()
