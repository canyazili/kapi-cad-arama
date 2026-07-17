# -*- coding: utf-8 -*-
"""core.py — Arama uygulamasının ortak yardımcıları (arayüzden bağımsız).

Flet arayüzü (kapi_arama_flet.py) buradan şu çekirdeği kullanır:
  - _setup_portable_caches: paketlenmiş exe'nin yanındaki 'modeller' önbelleğine yönlendirme
  - cad_image_path: sonuç çizim görselinin yolu
  - load_base_embeddings / group_results: görsel ikizleri gruplama
  - selftest: arayüz açmadan tam pipeline'ı koşan kendi kendini test

(Eski adı kapi_arama_app.py idi ve içinde artık kullanılmayan bir Tkinter arayüzü
vardı; Tkinter arayüzü kaldırıldı, dosya işlevine uygun şekilde core.py yapıldı.)

Kendi kendini test: KapiArama.exe --selftest [foto_yolu]
  (arayüz açmadan tam pipeline'ı koşar, sonucu selftest_sonuc.txt'ye yazar)
"""
import os
import sys
import traceback
from pathlib import Path

# PyInstaller --noconsole modunda stdout/stderr None olur; kütüphanelerin
# print/tqdm çağrıları patlamasın diye boş akışa bağlanır.
if sys.stdout is None:
    sys.stdout = open(Path.home() / ".kapi_arama_stdout.log", "a", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = sys.stdout


def _setup_portable_caches():
    """Taşınabilir kurulum: exe'nin yanında 'modeller' klasörü varsa model
    önbellekleri (DINOv2, HED, easyocr) oraya yönlendirilir — hedef PC'de
    internet/indirme gerekmez. Geliştirme makinesinde klasör yoksa dokunmaz."""
    if not getattr(sys, "frozen", False):
        return
    m = Path(sys.executable).resolve().parent / "modeller"
    if (m / "torch_hub").exists():
        os.environ.setdefault("TORCH_HOME", str(m / "torch_hub"))
    if (m / "hf").exists():
        os.environ.setdefault("HF_HOME", str(m / "hf"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # ağa çıkma, önbellekten al
    if (m / "easyocr").exists():
        os.environ.setdefault("EASYOCR_MODULE_PATH", str(m / "easyocr"))


_setup_portable_caches()

import search  # noqa: E402  (ROOT çözümü ve arama çekirdeği)

ROOT = search.ROOT
PHOTO_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")
DUP_SIM = 0.95       # ham DINOv2 uzayında "aynı tasarım" eşiği (etiketlerle kalibre)
DUP_FETCH_CAP = 150  # gruplarken indeksten istenecek en fazla ham sonuç


def cad_image_path(name: str) -> Path:
    """Gösterim için temiz CAD görseli; yoksa orijinal PNG."""
    clean = ROOT / "data" / "cad_clean" / name
    return clean if clean.exists() else ROOT / "cad_png" / name


_base_emb = None  # (embeddings, ad->satır) — ikiz gruplama için, tembel yüklenir


def load_base_embeddings():
    """Ham DINOv2 CAD embedding'leri (index/). Dosyalar yoksa None (gruplama kapanır)."""
    global _base_emb
    if _base_emb is None:
        import json
        import numpy as np
        emb_path = ROOT / "index" / "cad_embeddings.npy"
        names_path = ROOT / "index" / "cad_filenames.json"
        if not emb_path.exists() or not names_path.exists():
            _base_emb = (None, None)
        else:
            emb = np.load(emb_path).astype("float32")
            with open(names_path, "r", encoding="utf-8") as f:
                pos = {n: i for i, n in enumerate(json.load(f))}
            _base_emb = (emb, pos)
    return _base_emb


def group_results(results, k):
    """Sonuçları görsel ikizlerine göre tekilleştirir: skor sırasıyla gezilir,
    önceki bir temsilciye ham-benzerliği >= DUP_SIM olan sonuç onun varyantı
    olur; olmayan yeni temsilci açar (en fazla k temsilci).
    Dönüş: [(ad, skor, [(varyant_ad, varyant_skor), ...])]."""
    emb, pos = load_base_embeddings()
    if emb is None:
        return [(n, s, []) for n, s in results[:k]]
    reps = []  # [ad, skor, varyantlar, vektör]
    for name, score in results:
        v = emb[pos[name]] if name in pos else None
        placed = False
        if v is not None:
            for rep in reps:
                if rep[3] is not None and float(rep[3] @ v) >= DUP_SIM:
                    rep[2].append((name, score))
                    placed = True
                    break
        if not placed and len(reps) < k:
            reps.append([name, score, [], v])
    return [(r[0], r[1], r[2]) for r in reps]


def selftest(photo_arg: str = None) -> int:
    out = ROOT / "selftest_sonuc.txt"
    try:
        photo = Path(photo_arg) if photo_arg else next(
            p for pat in PHOTO_EXTS for p in sorted((ROOT / "photos").glob(pat)))
        engine = search.get_engine()
        results = engine.search(photo, k=5)
        lines = [f"FOTO: {photo}"] + [f"  {s:.4f}  {n}" for n, s in results]
        grouped = group_results(engine.search(photo, k=30), 5)
        lines.append("GRUPLAMA: " + ("aktif" if load_base_embeddings()[0] is not None
                                     else "kapalı (base embedding yok)"))
        lines += [f"  {s:.4f}  {n}  (+{len(v)} varyant)" for n, s, v in grouped]
        out.write_text("\n".join(lines) + "\nSELFTEST OK\n", encoding="utf-8")
        return 0
    except Exception:
        out.write_text(traceback.format_exc(), encoding="utf-8")
        return 1
