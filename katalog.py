# -*- coding: utf-8 -*-
"""katalog.py — Kataloğa çalışma anında yeni kapı (CAD çizimi) ekleme.

Uygulamadaki "Yeni Kapı Ekle" özelliğinin çekirdeği. Yeni çizim:
  1) 01_clean_cad.normalize_cad ile standart forma getirilir, data/cad_clean'e yazılır
     (cad_png/ salt-okunur kuralına dokunulmaz; küçük resimler cad_clean'den okunur),
  2) ham DINOv2 embedding'i base indekse (index/) eklenir — ikiz gruplama bunu kullanır,
  3) config'teki füzyon varyantlarının her birine kendi omurgası + projection.npz'siyle
     embedlenip eklenir (cad_embeddings.npy, cad_filenames.json, faiss.index),
  4) fotoğraf da verildiyse data/eklenen_fotolar'a kopyalanır ve foto↔çizim eşleşmesi
     data/eval/labels_ekli.json'a yazılır (ileride yeniden eğitimde kullanılmak üzere;
     insan-onaylı labels_clean.json'a ve donmuş ayrıma DOKUNULMAZ),
  5) kayıt index/eklenen_kapilar.json'a işlenir,
  6) engine verildiyse bellekteki arama da anında güncellenir (yeniden yükleme yok).

Tüm npy/json/faiss yazımları önce geçici dosyaya yapılır, sonra yer değiştirilir
(yarıda kesilirse indeks bozulmasın).
"""
import importlib.util
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np

import search

ROOT = search.ROOT
ADDED_PHOTOS_DIR = ROOT / "data" / "eklenen_fotolar"
ADDED_LABELS = ROOT / "data" / "eval" / "labels_ekli.json"
ADDED_LOG = ROOT / "index" / "eklenen_kapilar.json"

_clean_mod = None


def _load_clean_module():
    """scripts/01_clean_cad.py sayıyla başladığı için importlib ile yüklenir."""
    global _clean_mod
    if _clean_mod is None:
        path = ROOT / "scripts" / "01_clean_cad.py"
        spec = importlib.util.spec_from_file_location("clean_cad", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _clean_mod = mod
    return _clean_mod


def _replace_write(path: Path, write_fn):
    """Önce .tmp'ye yaz, sonra atomik değiştir (yarıda kesilme indeksi bozmasın)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    tmp.replace(path)


def _append_index_dir(index_dir: Path, new_names: list, vecs: np.ndarray):
    """Bir indeks klasörüne (embeddings + adlar + faiss) yeni satırlar ekler."""
    import faiss
    vecs = np.atleast_2d(vecs).astype(np.float32)
    emb_path = index_dir / "cad_embeddings.npy"
    names_path = index_dir / "cad_filenames.json"
    emb = np.load(emb_path).astype(np.float32)
    with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if vecs.shape[1] != emb.shape[1]:
        raise ValueError(f"boyut uyuşmazlığı: {vecs.shape[1]} != {emb.shape[1]} ({index_dir})")
    emb = np.vstack([emb, vecs])
    names.extend(new_names)

    def _save_npy(p):
        with open(p, "wb") as fh:   # np.save yola .npy eklemesin diye handle ile
            np.save(fh, emb)
    _replace_write(emb_path, _save_npy)
    _replace_write(names_path, lambda p: p.write_text(
        json.dumps(names, ensure_ascii=False), encoding="utf-8"))
    faiss_path = index_dir / "faiss.index"
    if faiss_path.exists():
        idx = faiss.deserialize_index(np.fromfile(faiss_path, dtype=np.uint8))
        idx.add(vecs)
        _replace_write(faiss_path, lambda p: p.write_bytes(
            faiss.serialize_index(idx).tobytes()))


def _sanitize_name(name: str) -> str:
    name = name.strip()
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    if not name.lower().endswith(".png"):
        name += ".png"
    return name


def list_entries():
    """Uygulamadan eklenen kapıların günlüğü (yeniden eskiye sıralı)."""
    if not ADDED_LOG.exists():
        return []
    with open(ADDED_LOG, "r", encoding="utf-8") as f:
        return list(reversed(json.load(f)))


def _remove_from_index_dir(index_dir: Path, names_to_remove: set):
    """Bir indeks klasöründen verilen adların satırlarını çıkarır."""
    import faiss
    emb_path = index_dir / "cad_embeddings.npy"
    names_path = index_dir / "cad_filenames.json"
    emb = np.load(emb_path).astype(np.float32)
    with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    keep = [i for i, n in enumerate(names) if n not in names_to_remove]
    if len(keep) == len(names):
        return 0
    removed = len(names) - len(keep)
    emb = emb[keep]
    names = [names[i] for i in keep]

    def _save_npy(p):
        with open(p, "wb") as fh:
            np.save(fh, emb)
    _replace_write(emb_path, _save_npy)
    _replace_write(names_path, lambda p: p.write_text(
        json.dumps(names, ensure_ascii=False), encoding="utf-8"))
    faiss_path = index_dir / "faiss.index"
    if faiss_path.exists():
        idx = faiss.IndexFlatIP(emb.shape[1])   # az satır silmek için baştan kur
        idx.add(emb)
        _replace_write(faiss_path, lambda p: p.write_bytes(
            faiss.serialize_index(idx).tobytes()))
    return removed


def remove_entry(entry, engine=None, log=print):
    """Uygulamadan eklenmiş bir kaydı TÜMÜYLE geri alır: çizimler indekslerden,
    fotoğraflar ve eşleşme kayıtları diskten silinir. Sadece eklenen_kapilar
    günlüğündeki kayıtlar silinebilir (orijinal kataloğa dokunulamaz)."""
    cfg = search.load_config()
    names = set(entry.get("cizimler", []))
    photo_names = set(entry.get("fotolar", []))
    if not names:
        raise ValueError("Kayıtta silinecek çizim yok.")

    # Güvence: günlükte birebir bu kayıt var mı?
    log_data = []
    if ADDED_LOG.exists():
        with open(ADDED_LOG, "r", encoding="utf-8") as f:
            log_data = json.load(f)
    hedef = [e for e in log_data
             if set(e.get("cizimler", [])) == names
             and set(e.get("fotolar", [])) == photo_names]
    if not hedef:
        raise ValueError("Bu kayıt günlükte bulunamadı — yalnızca uygulamadan "
                         "eklenen kapılar silinebilir.")

    # 1) Çizimleri tüm indekslerden çıkar
    dirs = [ROOT / cfg["paths"]["index_dir"]]
    variants = list((cfg.get("fusion") or {}).get("variants", []))
    active = cfg.get("active_index", "base")
    if active not in ("base", "fusion", None, "") and active not in variants:
        variants.append(active)
    dirs += [search.resolve_index_dir(cfg, v) for v in variants]
    for d in dirs:
        n = _remove_from_index_dir(d, names)
        log(f"İndeksten çıkarıldı ({d.name}): {n} çizim")

    # 2) cad_clean kopyaları ve eklenen fotoğraflar
    for nm in names:
        p = ROOT / "data" / "cad_clean" / nm
        if p.exists():
            p.unlink()
    for ph in photo_names:
        p = ADDED_PHOTOS_DIR / ph
        if p.exists():
            p.unlink()

    # 3) Eşleşme kayıtları
    if ADDED_LABELS.exists():
        with open(ADDED_LABELS, "r", encoding="utf-8") as f:
            labels = json.load(f)
        es = labels.get("eslesme", {})
        for ph in photo_names:
            es.pop(ph, None)
        _replace_write(ADDED_LABELS, lambda p: p.write_text(
            json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8"))

    # 4) Günlükten düş
    log_data = [e for e in log_data
                if not (set(e.get("cizimler", [])) == names
                        and set(e.get("fotolar", [])) == photo_names)]
    _replace_write(ADDED_LOG, lambda p: p.write_text(
        json.dumps(log_data, ensure_ascii=False, indent=1), encoding="utf-8"))

    # 5) Bellekteki engine'den düş (varsa)
    if engine is not None and getattr(engine, "fusion_sides", None):
        keep = [i for i, n in enumerate(engine.filenames) if n not in names]
        if len(keep) != len(engine.filenames):
            for s in engine.fusion_sides:
                s["emb"] = s["emb"][keep]
            engine.filenames = [engine.filenames[i] for i in keep]

    msg = f"Silindi: {len(names)} çizim + {len(photo_names)} fotoğraf"
    log("TAMAM: " + msg)
    return msg


def remove_drawing(entry, drawing_name, engine=None, log=print):
    """Eklenmiş bir kayıttan TEK çizimi geri alır; fotoğraflar ve diğer çizimler
    kalır. Kayıttaki son çizimse (fotoğraflar eşleşmesiz kalacağından) kaydın
    tamamı silinir (remove_entry)."""
    cizimler = list(entry.get("cizimler", []))
    if drawing_name not in cizimler:
        raise ValueError(f"'{drawing_name}' bu kayıtta yok.")
    if len(cizimler) == 1:
        log("Kayıttaki son çizim — ekleme komple geri alınıyor.")
        return remove_entry(entry, engine=engine, log=log)

    cfg = search.load_config()
    # Günlükte birebir kaydı bul
    log_data = []
    if ADDED_LOG.exists():
        with open(ADDED_LOG, "r", encoding="utf-8") as f:
            log_data = json.load(f)
    hedef = None
    for e in log_data:
        if (set(e.get("cizimler", [])) == set(cizimler)
                and set(e.get("fotolar", [])) == set(entry.get("fotolar", []))):
            hedef = e
            break
    if hedef is None:
        raise ValueError("Bu kayıt günlükte bulunamadı — yalnızca uygulamadan "
                         "eklenen çizimler silinebilir.")

    # 1) Çizimi tüm indekslerden çıkar
    dirs = [ROOT / cfg["paths"]["index_dir"]]
    variants = list((cfg.get("fusion") or {}).get("variants", []))
    active = cfg.get("active_index", "base")
    if active not in ("base", "fusion", None, "") and active not in variants:
        variants.append(active)
    dirs += [search.resolve_index_dir(cfg, v) for v in variants]
    for d in dirs:
        _remove_from_index_dir(d, {drawing_name})
    log(f"İndekslerden çıkarıldı: {drawing_name}")

    # 2) cad_clean kopyası
    p = ROOT / "data" / "cad_clean" / drawing_name
    if p.exists():
        p.unlink()

    # 3) Eşleşme kayıtlarından bu çizimi düş (fotolar kalır)
    if ADDED_LABELS.exists():
        with open(ADDED_LABELS, "r", encoding="utf-8") as f:
            labels = json.load(f)
        for ph in entry.get("fotolar", []):
            labels.get("eslesme", {}).get(ph, {}).pop(drawing_name, None)
        _replace_write(ADDED_LABELS, lambda p_: p_.write_text(
            json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8"))

    # 4) Günlükteki kaydı güncelle
    hedef["cizimler"] = [c for c in hedef["cizimler"] if c != drawing_name]
    _replace_write(ADDED_LOG, lambda p_: p_.write_text(
        json.dumps(log_data, ensure_ascii=False, indent=1), encoding="utf-8"))

    # 5) Bellekteki engine'den düş (varsa)
    if engine is not None and getattr(engine, "fusion_sides", None):
        keep = [i for i, n in enumerate(engine.filenames) if n != drawing_name]
        if len(keep) != len(engine.filenames):
            for s in engine.fusion_sides:
                s["emb"] = s["emb"][keep]
            engine.filenames = [engine.filenames[i] for i in keep]

    msg = f"Silindi: {drawing_name} (kayıtta {len(hedef['cizimler'])} çizim kaldı)"
    log("TAMAM: " + msg)
    return msg


def add_entry(drawing_paths, photo_paths, engine=None, log=print):
    """Yeni kapıyı kataloğa ekler: en az bir ÇİZİM ve en az bir FOTOĞRAF zorunlu
    (çizimsiz kapı / kapısız çizim eklenemez — eşleşme kaydı hep tam olsun).

    drawing_paths: çizim görselleri (AutoCAD'den PNG/JPG export) — hepsi indekse girer,
                   katalog adları dosya adlarından otomatik gelir.
    photo_paths:   kapının gerçek fotoğrafları — kopyalanır, her foto↔her çizim
                   eşleşmesi labels_ekli.json'a yazılır (gelecek eğitimler için).
    engine: açık SearchEngine (füzyon modu) — verilirse bellekte de güncellenir
            ve embedder'ları yeniden yüklemeye gerek kalmaz.
    Özet metni döner; hatada exception fırlatır.
    """
    from PIL import Image

    cfg = search.load_config()
    drawings = [Path(p) for p in drawing_paths]
    photos = [Path(p) for p in photo_paths]
    if not drawings or not photos:
        raise ValueError("En az bir çizim VE bir fotoğraf gerekli.")

    # Adlar çizim dosya adlarından; çakışma varsa hiçbir şey yazmadan dur
    with open(ROOT / "index" / "cad_filenames.json", "r", encoding="utf-8") as f:
        existing = set(json.load(f))
    names = []
    for dp in drawings:
        nm = _sanitize_name(dp.stem)
        if nm in existing or nm in names:
            raise ValueError(f"'{nm}' zaten katalogda var — çizim dosyasını yeniden adlandırın.")
        names.append(nm)

    # 1) Önce TÜM çizimleri normalize et (biri bozuksa hiçbiri eklenmesin);
    #    DWG/DXF dosyaları önce PNG'ye render edilir (dwg2png)
    cleans = []
    for dp, nm in zip(drawings, names):
        if dp.suffix.lower() in (".dwg", ".dxf"):
            log(f"AutoCAD çizimi PNG'ye çevriliyor: {dp.name}")
            import dwg2png
            dp = dwg2png.cad_to_png(dp)
        log(f"Çizim normalize ediliyor: {dp.name} -> {nm}")
        with Image.open(dp) as im:
            im.load()
            cleans.append(_load_clean_module().normalize_cad(im, cfg["image_size"]))
    clean_dir = ROOT / "data" / "cad_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    for nm, cl in zip(names, cleans):
        cl.save(clean_dir / nm)

    # Embedder havuzu: engine'inkiler varsa onları kullan (model tekrar yüklenmesin)
    embedders = dict(getattr(engine, "fusion_embedders", {}) or {})

    def get_embedder(model_name):
        if model_name not in embedders:
            log(f"Model yükleniyor: {model_name}")
            embedders[model_name] = search.make_embedder(cfg, model_name=model_name)
        return embedders[model_name]

    # 2) Base indeks (ham DINOv2) — ikiz gruplama ve base arama bunu kullanır
    base_model = cfg["model"]["name"]
    log("Embedding hesaplanıyor (base)…")
    raw_by_model = {base_model: get_embedder(base_model).embed_batch(cleans)}
    _append_index_dir(ROOT / cfg["paths"]["index_dir"], names, raw_by_model[base_model])

    # 3) Füzyon varyantları (+ aktif tekil varyant, füzyon dışındaysa)
    variants = list((cfg.get("fusion") or {}).get("variants", []))
    active = cfg.get("active_index", "base")
    if active not in ("base", "fusion", None, "") and active not in variants:
        variants.append(active)
    projected_by_variant = {}
    for v in variants:
        d = search.resolve_index_dir(cfg, v)
        model = cfg["model"]["name"]
        meta_path = d / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                model = json.load(f).get("model", model)
        if model not in raw_by_model:
            log(f"Embedding hesaplanıyor ({model})…")
            raw_by_model[model] = get_embedder(model).embed_batch(cleans)
        vecs = raw_by_model[model]
        proj_path = d / "projection.npz"
        if proj_path.exists():
            vecs = search.NpzProjection(proj_path)(vecs).astype(np.float32)
        _append_index_dir(d, names, vecs)
        projected_by_variant[v] = vecs
        log(f"İndekse eklendi: {v}")

    # 4) Fotoğrafları kopyala + her foto↔her çizim eşleşmesini kaydet
    labels = {"eslesme": {}}
    if ADDED_LABELS.exists():
        with open(ADDED_LABELS, "r", encoding="utf-8") as f:
            labels = json.load(f)
    ADDED_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    photo_names = []
    for pp in photos:
        dst = ADDED_PHOTOS_DIR / pp.name
        i = 1
        while dst.exists():
            dst = ADDED_PHOTOS_DIR / f"{pp.stem}_{i}{pp.suffix}"
            i += 1
        shutil.copy2(pp, dst)
        entry = labels.setdefault("eslesme", {}).setdefault(dst.name, {})
        for nm in names:
            entry[nm] = "app"
        photo_names.append(dst.name)
    _replace_write(ADDED_LABELS, lambda p: p.write_text(
        json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8"))

    # 5) Ekleme günlüğü
    log_data = []
    if ADDED_LOG.exists():
        with open(ADDED_LOG, "r", encoding="utf-8") as f:
            log_data = json.load(f)
    log_data.append({"cizimler": names, "fotolar": photo_names,
                     "tarih": str(date.today())})
    _replace_write(ADDED_LOG, lambda p: p.write_text(
        json.dumps(log_data, ensure_ascii=False, indent=1), encoding="utf-8"))

    # 6) Bellekteki engine'i güncelle (varsa) — arama anında yeni kapıyı görür
    if engine is not None and getattr(engine, "fusion_sides", None):
        fus = (cfg.get("fusion") or {})
        for s, v in zip(engine.fusion_sides, fus.get("variants", [])):
            s["emb"] = np.vstack([s["emb"], projected_by_variant[v]])
        engine.filenames.extend(names)

    msg = (f"{len(names)} çizim + {len(photo_names)} fotoğraf eklendi: "
           f"{', '.join(names)}")
    log("TAMAM: " + msg)
    return msg
