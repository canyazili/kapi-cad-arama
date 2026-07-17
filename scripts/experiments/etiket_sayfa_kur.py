# -*- coding: utf-8 -*-
"""
etiket_sayfa_kur.py — Tek-tek gezilen etiketleme sayfasını kurar.

Girdi : data/eval/etiketleme/oneriler.json (etiket_oneri_uret.py çıktısı)
Çıktı : data/eval/etiketleme/etiketle.html + veri.js

Sayfa özellikleri:
  - Bir ekranda TEK fotoğraf; ◀/▶ veya klavye okları ile sırayla gezilir
  - Model önerileri skorlu kartlar halinde; tıklayınca seçilir/bırakılır
  - "Çizim ara" kutusu: isim yazınca 32 bin çizim içinde anında arama
  - "Eşleşme yok" işareti; işaretler tarayıcıda saklanır (localStorage)
  - "İşaretleri indir" -> etiket_onay.json; uygulamak için:
        python scripts/experiments/etiket_uygula.py etiket_onay.json

Kullanım: python scripts/experiments/etiket_sayfa_kur.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from label_tools import normalize_name  # noqa: E402

OUT_DIR = ROOT / "data" / "eval" / "etiketleme"

_PAGE = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kapı Etiketleme — tek tek</title>
<style>
 *{box-sizing:border-box}
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#eef2f7;color:#1a202c}
 header{position:sticky;top:0;z-index:9;background:#1f3a5f;color:#fff;
        padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 header b{color:#7dd3fc}
 button{cursor:pointer;border:0;border-radius:6px;padding:8px 14px;font-size:14px;background:#e2e8f0}
 .nav{background:#2b6cb0;color:#fff;font-weight:600}
 #btn-export{background:#059669;color:#fff}
 #btn-reset{background:#475569;color:#fff}
 #jump{width:64px;padding:6px;border-radius:6px;border:0;text-align:center}
 .hint{font-size:12px;color:#cbd5e1}
 main{display:flex;gap:14px;padding:14px;align-items:flex-start}
 .card{background:#fff;border:1px solid #d8dfea;border-radius:10px;padding:12px}
 #sol{flex:0 0 380px;position:sticky;top:64px;text-align:center}
 #sol img{max-width:100%;max-height:520px;border:1px solid #ccc;background:#fff;cursor:zoom-in}
 #foto-ad{font-size:13px;word-break:break-all;margin-top:6px;color:#333;font-weight:600}
 #btn-yok{margin-top:10px;width:100%}
 #btn-yok.on{background:#c0392b;color:#fff}
 #sag{flex:1;min-width:0}
 h3{margin:4px 0 8px;font-size:15px;color:#1f3a5f}
 .grid{display:flex;flex-wrap:wrap;gap:10px}
 .cell{width:150px;text-align:center;border:2px solid #ddd;border-radius:8px;
       padding:6px;cursor:pointer;background:#fff}
 .cell img{max-width:136px;max-height:170px;background:#fff}
 .cell .ad{font-size:10px;word-break:break-all;color:#444;min-height:24px}
 .cell .skor{display:inline-block;font-size:10px;background:#0369a1;color:#fff;
             border-radius:4px;padding:1px 6px;margin-bottom:2px}
 .cell.secili{border-color:#059669;background:#ecfdf5;box-shadow:0 0 0 2px #05966944}
 #secilenler{display:flex;flex-wrap:wrap;gap:6px;min-height:30px;margin-bottom:10px}
 .chip{background:#059669;color:#fff;border-radius:14px;padding:4px 10px;font-size:12px;
       display:flex;align-items:center;gap:6px}
 .chip span{cursor:pointer;font-weight:700}
 #ara{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:8px;font-size:15px}
 #ara-sonuc{margin-top:10px}
 .bos{color:#64748b;font-size:13px;padding:8px 0}
 .durum-nokta{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:4px}
 #onizleme{position:fixed;inset:0;background:#000c;display:none;z-index:99;
           align-items:center;justify-content:center;cursor:zoom-out}
 #onizleme img{max-width:96vw;max-height:96vh;background:#fff}
</style></head><body>
<header>
 <button class="nav" id="btn-prev">◀ Önceki</button>
 <span>Foto <b id="poz">1</b> / <b id="toplam">?</b></span>
 <button class="nav" id="btn-next">Sonraki ▶</button>
 <button id="btn-bos">Sonraki işaretsiz ⏭</button>
 <input id="jump" type="number" min="1" title="Numaraya git">
 <span style="margin-left:8px"><span class="durum-nokta" style="background:#059669"></span><b id="n-etiketli">0</b> etiketli</span>
 <span><span class="durum-nokta" style="background:#c0392b"></span><b id="n-yok">0</b> eşleşme yok</span>
 <span><span class="durum-nokta" style="background:#94a3b8"></span><b id="n-kalan">0</b> kaldı</span>
 <button id="btn-export">İşaretleri indir (JSON)</button>
 <button id="btn-reset">Sıfırla</button>
 <span class="hint">← → tuşlarıyla gezin · işaretler tarayıcıda saklanır · fotoğrafa tıkla = büyüt</span>
</header>
<main>
 <div id="sol" class="card">
   <img id="foto" alt="">
   <div id="foto-ad"></div>
   <button id="btn-yok">🚫 Eşleşme yok / atla</button>
 </div>
 <div id="sag">
   <div class="card" style="margin-bottom:12px">
     <h3>Seçilen çizimler</h3>
     <div id="secilenler"><span class="bos">Henüz seçim yok</span></div>
   </div>
   <div class="card" style="margin-bottom:12px">
     <h3>🤖 Model önerileri</h3>
     <div id="oneriler" class="grid"></div>
   </div>
   <div class="card">
     <h3>🔎 Çizim ara (isim yaz)</h3>
     <input id="ara" placeholder="örn: 1008, halaskar, fu2614..." autocomplete="off">
     <div id="ara-sonuc" class="grid"></div>
   </div>
 </div>
</main>
<div id="onizleme"><img id="onizleme-img"></div>
<script src="veri.js"></script>
<script>
const KEY = "etiketle:v1";
const store = JSON.parse(localStorage.getItem(KEY) || "{}");
function kaydet(){ localStorage.setItem(KEY, JSON.stringify(store)); sayac(); }

const F = VERI.fotolar, CADS = VERI.cadlar;   // CADS: [ad, normalize] listesi
let idx = parseInt(localStorage.getItem(KEY+":poz") || "0");
if (isNaN(idx) || idx < 0 || idx >= F.length) idx = 0;

const TRMAP = {"ö":"o","ü":"u","ş":"s","ç":"c","ğ":"g","ı":"i","İ":"i","Ö":"o","Ü":"u","Ş":"s","Ç":"c","Ğ":"g"};
function norm(s){ return s.replace(/[öüşçğıİÖÜŞÇĞ]/g, c=>TRMAP[c]||c).toLowerCase().replace(/\s+/g,""); }

function cadSrc(ad){ return "../../cad_clean/" + encodeURIComponent(ad); }
function kayit(foto){ return store[foto] || (store[foto] = {sec:{}, yok:false}); }

function sayac(){
  let et=0, yok=0;
  for(const f of F){
    const r = store[f.ad];
    if(!r) continue;
    if(Object.keys(r.sec).length) et++;
    else if(r.yok) yok++;
  }
  document.getElementById("n-etiketli").textContent = et;
  document.getElementById("n-yok").textContent = yok;
  document.getElementById("n-kalan").textContent = F.length - et - yok;
}

function hucre(ad, skor, tur){
  const r = kayit(F[idx].ad);
  const d = document.createElement("div");
  d.className = "cell" + (r.sec[ad] ? " secili" : "");
  d.innerHTML = (skor!=null?`<div class="skor">${skor.toFixed(2)}</div>`:"") +
    `<img loading="lazy" src="${cadSrc(ad)}"><div class="ad">${ad}</div>`;
  d.onclick = ()=>{
    const r2 = kayit(F[idx].ad);
    if(r2.sec[ad]) delete r2.sec[ad];
    else { r2.sec[ad] = tur; r2.yok = false; }
    kaydet(); goster();
  };
  return d;
}

function goster(){
  localStorage.setItem(KEY+":poz", idx);
  const f = F[idx], r = kayit(f.ad);
  document.getElementById("poz").textContent = idx+1;
  document.getElementById("jump").value = idx+1;
  const img = document.getElementById("foto");
  img.src = "thumbs/" + encodeURIComponent(f.thumb);
  img.onclick = ()=>{
    document.getElementById("onizleme-img").src = "../../../photos/" + encodeURIComponent(f.ad);
    document.getElementById("onizleme").style.display = "flex";
  };
  document.getElementById("foto-ad").textContent = f.ad;
  const by = document.getElementById("btn-yok");
  by.className = r.yok ? "on" : "";

  // seçilenler
  const sec = document.getElementById("secilenler");
  sec.innerHTML = "";
  const adlar = Object.keys(r.sec);
  if(!adlar.length) sec.innerHTML = '<span class="bos">Henüz seçim yok</span>';
  for(const ad of adlar){
    const c = document.createElement("div"); c.className = "chip";
    c.innerHTML = `${ad} <span title="çıkar">✕</span>`;
    c.querySelector("span").onclick = ()=>{ delete r.sec[ad]; kaydet(); goster(); };
    sec.appendChild(c);
  }

  // öneriler
  const on = document.getElementById("oneriler");
  on.innerHTML = "";
  if(!f.oneri.length) on.innerHTML = '<span class="bos">Bu foto için öneri üretilemedi — aramayı kullanın</span>';
  for(const [ad, skor] of f.oneri) on.appendChild(hucre(ad, skor, "oneri"));

  ara();  // arama sonuçlarının seçili durumunu tazele
  sayac();
  window.scrollTo(0,0);
}

function git(n){ idx = Math.min(Math.max(n,0), F.length-1); goster(); }

function ara(){
  const q = norm(document.getElementById("ara").value.trim());
  const kutu = document.getElementById("ara-sonuc");
  kutu.innerHTML = "";
  if(q.length < 2){ kutu.innerHTML = '<span class="bos">En az 2 karakter yazın</span>'; return; }
  let n = 0;
  for(const [ad, nrm] of CADS){
    if(nrm.includes(q)){
      kutu.appendChild(hucre(ad, null, "arama"));
      if(++n >= 60){
        const s = document.createElement("span"); s.className="bos";
        s.textContent = "… 60'tan fazla sonuç, aramayı daraltın";
        kutu.appendChild(s); break;
      }
    }
  }
  if(!n) kutu.innerHTML = '<span class="bos">Eşleşen çizim yok</span>';
}

document.getElementById("btn-prev").onclick = ()=>git(idx-1);
document.getElementById("btn-next").onclick = ()=>git(idx+1);
document.getElementById("btn-bos").onclick = ()=>{
  for(let i=1;i<=F.length;i++){
    const j=(idx+i)%F.length, r=store[F[j].ad];
    if(!r || (!Object.keys(r.sec).length && !r.yok)){ git(j); return; }
  }
  alert("Tebrikler, işaretsiz foto kalmadı!");
};
document.getElementById("btn-yok").onclick = ()=>{
  const r = kayit(F[idx].ad);
  r.yok = !r.yok;
  if(r.yok) r.sec = {};
  kaydet(); goster();
};
document.getElementById("jump").onchange = e=>git(parseInt(e.target.value||"1")-1);
document.getElementById("ara").oninput = ara;
document.getElementById("onizleme").onclick = ()=>{ document.getElementById("onizleme").style.display="none"; };
document.addEventListener("keydown", e=>{
  if(e.target.tagName === "INPUT") return;
  if(e.key === "ArrowLeft") git(idx-1);
  if(e.key === "ArrowRight") git(idx+1);
  if(e.key === "Escape") document.getElementById("onizleme").style.display="none";
});
document.getElementById("btn-reset").onclick = ()=>{
  if(confirm("TÜM işaretler silinsin mi? (indirmediyseniz kaybolur)")){
    for(const k in store) delete store[k];
    kaydet(); goster();
  }
};
document.getElementById("btn-export").onclick = ()=>{
  const onay={}, tur={}, yok=[];
  for(const [foto, r] of Object.entries(store)){
    const adlar = Object.keys(r.sec||{});
    if(adlar.length){ onay[foto]=adlar; tur[foto]=r.sec; }
    else if(r.yok) yok.push(foto);
  }
  const blob = new Blob([JSON.stringify({onay, tur, yok}, null, 1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "etiket_onay.json"; a.click();
};

document.getElementById("toplam").textContent = F.length;
goster();
</script></body></html>
"""


def main():
    oneriler = json.load(open(OUT_DIR / "oneriler.json", encoding="utf-8"))

    # atlananlar (önceki turlarda "eşleşme yok" denenler) sayfadan düşülür
    atla_path = OUT_DIR / "atlananlar.json"
    atlanan = set(json.load(open(atla_path, encoding="utf-8"))) if atla_path.exists() else set()

    labels = json.load(open(ROOT / "data/eval/labels_clean.json", encoding="utf-8"))
    labeled = set(labels["eslesme"])

    fotolar = []
    for ad in sorted(oneriler):
        if ad in labeled or ad in atlanan:
            continue
        v = oneriler[ad]
        fotolar.append({
            "ad": ad,
            "thumb": ad + ".jpg",
            "oneri": v if isinstance(v, list) else [],
        })

    cad_adlar = json.load(open(ROOT / "index/cad_filenames.json", encoding="utf-8"))
    cadlar = [[ad, normalize_name(Path(ad).stem)] for ad in sorted(set(cad_adlar))]

    veri = {"fotolar": fotolar, "cadlar": cadlar}
    with open(OUT_DIR / "veri.js", "w", encoding="utf-8") as f:
        f.write("const VERI = " + json.dumps(veri, ensure_ascii=False) + ";\n")
    (OUT_DIR / "etiketle.html").write_text(_PAGE, encoding="utf-8")
    print(f"Sayfa hazır: {OUT_DIR / 'etiketle.html'}")
    print(f"  {len(fotolar)} etiketsiz foto, {len(cadlar)} aranabilir çizim")


if __name__ == "__main__":
    main()
