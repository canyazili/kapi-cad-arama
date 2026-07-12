# -*- coding: utf-8 -*-
"""
07_train_projection.py — Donuk DINOv2 üstüne 2 katmanlı MLP projeksiyon eğitimi.

Amaç: fotoğraf-lineart (HED stili) ile CAD-clean embedding'lerini ortak bir
uzaya çeken 768->512 projeksiyon. DINOv2 DONUK kalır; sadece MLP eğitilir,
bu yüzden eğitim verisi = önceden hesaplanmış embedding'ler (hızlı).

Veri:
  a) Sentetik çiftler: her CAD'in cad_clean hali (index/cad_embeddings.npy) ile
     hedthick5 hali (index/variants/hedthick5_vitb14/...) pozitif çift (~32k).
     'prepare' aşaması ek olarak hafif augmentation'lı (±3° rotasyon, ölçek,
     perspektif) kopyaların embedding'lerini üretip data/train_cache/'e yazar.
  b) Gerçek çiftler: labels_clean.json'dan (foto lineart, doğru CAD) çiftleri.
     Fotoğraflar AİLE bazında (isim kökü, ör. BAŞARI1020) üçe ayrılır:
       test (~%25)      — SADECE kaynak etiketi manual/prefix_reviewed olan
                          ailelerden seçilir (exact_auto'da gürültü olabilir;
                          test temiz kalsın). Eğitimde ve erken durdurmada ASLA
                          kullanılmaz; eğitim bitince TEK KEZ ölçülür.
       validation (~%15 eğitim ailelerinden) — erken durdurma val-r@10 ile.
       eğitim (kalan)   — gerçek çiftler yalnız buradan.
     Aynı ailenin tüm fotoğrafları hep aynı tarafta (sızıntı yok).

Loss: simetrik InfoNCE (in-batch negatives; aynı CAD'e giden satırlar false
negative olmasın diye maskelenir), sentetik+gerçek karışık batch'ler, cosine LR.

Aşamalar:
  python scripts/07_train_projection.py prepare            # augment embedding cache (uzun, GPU)
  python scripts/07_train_projection.py train              # eğitim + rapor (cache varsa kullanır)
  python scripts/07_train_projection.py train --no-aug     # augment cache'i beklemeden eğit
  python scripts/07_train_projection.py build              # en iyi modelle index/variants/projected
  python scripts/07_train_projection.py all                # hepsi sırayla

Sonunda 04 ile bağımsız doğrulama (eğitim/val/test AYRI AYRI; fark = overfitting):
  python scripts/04_evaluate.py --variant projected --photos-file data/eval/projection_train_photos.json
  python scripts/04_evaluate.py --variant projected --photos-file data/eval/projection_val_photos.json
  python scripts/04_evaluate.py --variant projected --photos-file data/eval/projection_test_photos.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from search import DinoEmbedder, load_config, load_excluded_trivial, make_embedder, resolve_index_dir  # noqa: E402
from label_tools import photo_root  # noqa: E402

CACHE_DIR = ROOT / "data" / "train_cache"
MODELS_DIR = ROOT / "data" / "models"
EVAL_DIR = ROOT / "data" / "eval"
EMB_DIM, PROJ_DIM = 768, 512

# Omurga seçimi: CAD varyant çifti + sorgu/foto modeli + cache ayrımı.
# dinov3 foto embedding'leri embed_dinov3.py'nin npz'sinden okunur (varsa).
BACKBONES = {
    "dinov2": {"clean": "base", "hed": "hedthick5_vitb14", "model": "dinov2_vitb14",
               "photo_npz": None, "aug_prefix": ""},
    "dinov3": {"clean": "dinov3_base", "hed": "dinov3_hedthick5", "model": "dinov3_vitb16",
               "photo_npz": CACHE_DIR / "dinov3" / "photos_dinov3.npz",
               "aug_prefix": "d3_"},
}
TEMPERATURE = 0.07
TEST_FRACTION = 0.25   # tüm ailelerin ~%25'i (temiz kaynaklılardan seçilir)
VAL_FRACTION = 0.15    # kalan eğitim ailelerinin ~%15'i (erken durdurma)
CLEAN_TAGS = {"manual", "prefix_reviewed"}  # test kümesine girebilecek kaynaklar
SEED = 42


# ------------------------------------------------------------------ yardımcılar

def aligned_names(cfg, variants=("base", "hedthick5_vitb14")):
    """clean ve hed indekslerinin ortak dosya adları — sıralı."""
    names = []
    for variant in variants:
        d = resolve_index_dir(cfg, variant)
        with open(d / "cad_filenames.json", "r", encoding="utf-8") as f:
            names.append(json.load(f))
    common = sorted(set(names[0]) & set(names[1]))
    return common, names


def load_side(cfg, variant, common):
    """Bir varyantın embedding matrisini ortak ada göre hizalar. (N, 768)"""
    d = resolve_index_dir(cfg, variant)
    emb = np.load(d / "cad_embeddings.npy").astype(np.float32)
    with open(d / "cad_filenames.json", "r", encoding="utf-8") as f:
        pos = {n: i for i, n in enumerate(json.load(f))}
    return emb[[pos[n] for n in common]]


def augment_image(pil_img: Image.Image, rng: random.Random) -> Image.Image:
    """Hafif augmentation: %50 yatay çevirme + ±3° rotasyon + %5 ölçek +
    küçük perspektif (beyaz dolgu). Yatay çevirme: kapılar sağ/sol menteşeli
    üretilir, ayna kopya aynı model sayılır — projeksiyon flip-duyarsız olmalı."""
    arr = np.asarray(pil_img.convert("L"))
    if rng.random() < 0.5:
        arr = arr[:, ::-1].copy()
    h, w = arr.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), rng.uniform(0.95, 1.05))
    arr = cv2.warpAffine(arr, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
    j = 0.015 * min(w, h)  # perspektif köşe oynatması
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    P = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(arr, P, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
    return Image.fromarray(arr).convert("RGB")


def embed_folder(embedder, src_dir: Path, names, out_path: Path, augment: bool,
                 batch_size: int, seed: int):
    """Klasördeki 'names' görsellerini (istenirse augment ederek) embedler.
    03'teki gibi checkpoint'li: yarıda kesilirse kaldığı yerden devam eder."""
    ckpt = out_path.with_suffix(".part.npz")
    chunks, start = [], 0
    if ckpt.exists():
        try:
            d = np.load(ckpt, allow_pickle=False)
            done = [n for n in d["filenames"].tolist()]
            if done == names[:len(done)]:
                chunks, start = [d["embeddings"]], len(done)
                print(f"  checkpoint: {start} hazır, devam ediliyor")
        except Exception:
            pass
    pbar = tqdm(total=len(names), initial=start, desc=out_path.stem, unit="img")
    since = 0
    for i in range(start, len(names), batch_size):
        batch = []
        for n in names[i:i + batch_size]:
            with Image.open(src_dir / n) as im:
                im.load()
                img = im.convert("RGB")
            # görüntü başına ayrı tohum: checkpoint'ten devam edilse de aynı
            # augmentation üretilir (çekiliş sayısından bağımsız)
            batch.append(augment_image(img, random.Random(f"{seed}:{n}"))
                         if augment else img)
        chunks.append(embedder.embed_batch(batch))
        pbar.update(len(batch))
        since += 1
        if since >= 20:
            emb = np.concatenate(chunks, axis=0)
            chunks = [emb]
            np.savez(ckpt, embeddings=emb, filenames=np.array(names[:len(emb)]))
            since = 0
    pbar.close()
    emb = np.concatenate(chunks, axis=0).astype(np.float32)
    np.savez(out_path, embeddings=emb, filenames=np.array(names))
    if ckpt.exists():
        ckpt.unlink()
    print(f"  -> {out_path.relative_to(ROOT)} ({emb.shape})")


def load_labels_clean(allow_dirty: bool):
    """(eslesme, kaynak) döner; kaynak etiketi olmayan çiftler 'manual' sayılır."""
    path = EVAL_DIR / "labels_clean.json"
    if not path.exists():
        if not allow_dirty:
            print("UYARI: labels_clean.json yok. Önce 06/09 onay turlarını tamamlayın\n"
                  "ya da --allow-dirty-labels ile ham labels.json'la eğitin.")
            sys.exit(1)
        path = ROOT / "labels.json"
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    eslesme = data.get("eslesme", {})
    kaynak = data.get("kaynak", {})
    print(f"Etiketler: {path.name} ({len(eslesme)} kayıt)")
    return eslesme, kaynak


def family_split(photos, photo_tags):
    """Fotoğrafları isim kökü ailelerine göre üçe ayırır (sızıntı yok):
    test yalnız CLEAN_TAGS kaynaklı ailelerden, validation eğitim ailelerinden."""
    families = {}
    for p in photos:
        families.setdefault(photo_root(p), []).append(p)
    fam_names = sorted(families)
    # bir aile "temiz" = tüm fotoğraflarının tüm çift kaynakları CLEAN_TAGS içinde
    clean_fams = [f for f in fam_names
                  if all(photo_tags[p] <= CLEAN_TAGS for p in families[f])]
    rng = random.Random(SEED)

    n_test = max(1, round(len(fam_names) * TEST_FRACTION))
    if n_test > len(clean_fams):
        print(f"UYARI: temiz kaynaklı aile ({len(clean_fams)}) hedef test sayısından "
              f"({n_test}) az; test kümesi {len(clean_fams)} aileyle sınırlandı.")
        n_test = len(clean_fams)
    shuffled_clean = clean_fams[:]
    rng.shuffle(shuffled_clean)
    test_fams = set(shuffled_clean[:n_test])

    rest = [f for f in fam_names if f not in test_fams]
    rng.shuffle(rest)
    n_val = max(1, round(len(rest) * VAL_FRACTION))
    val_fams = set(rest[:n_val])

    train = sorted(p for f in rest[n_val:] for p in families[f])
    val = sorted(p for f in val_fams for p in families[f])
    test = sorted(p for f in test_fams for p in families[f])
    print(f"Aile ayrımı: {len(fam_names)} aile ({len(clean_fams)} temiz kaynaklı) -> "
          f"eğitim {len(train)} foto ({len(rest) - n_val} aile), "
          f"val {len(val)} foto ({n_val} aile), "
          f"test {len(test)} foto ({n_test} aile, hepsi temiz)")
    return train, val, test


def frozen_split(photos, prev):
    """Önceki ayrımı (projection_split.json) DONDURUR: val/test aynen korunur —
    hem önceki sonuçlarla karşılaştırılabilirlik hem sızıntı güvenliği için
    (model_reviewed etiketler eski modelin tahmini; ayrım kayarsa eski eğitim
    aileleri yeni teste düşüp metriği şişirebilir). Yeni fotoğraflar eğitime
    eklenir; ailesi val/test'te olan yeni fotoğraflar TAMAMEN dışarıda kalır."""
    prev_assign = {}
    for split_name in ("train", "val", "test"):
        for p in prev[split_name]:
            prev_assign[p] = split_name
    root_split = {photo_root(p): s for p, s in prev_assign.items()}
    photo_set = set(photos)
    train = [p for p in prev["train"] if p in photo_set]
    val = [p for p in prev["val"] if p in photo_set]
    test = [p for p in prev["test"] if p in photo_set]
    n_old_train = len(train)
    dropped = []
    for p in photos:
        if p in prev_assign:
            continue
        if root_split.get(photo_root(p)) in ("val", "test"):
            dropped.append(p)
        else:
            train.append(p)
    train.sort()
    print(f"DONMUŞ ayrım: eğitim {len(train)} foto ({len(train) - n_old_train} yeni), "
          f"val {len(val)}, test {len(test)} (öncekiyle aynı); "
          f"{len(dropped)} yeni foto ailesi val/test'te olduğu için dışarıda (sızıntı önlendi). "
          f"Sıfırdan ayrım için: --resplit")
    return train, val, test


def collect_photo_embeddings_bk(bk, photo_names, cfg):
    """Omurgaya göre foto embedding'leri: önce npz cache (dinov3), eksikler
    ilgili modelle embedlenir ve npz'ye geri yazılır. dinov2'de npz yok,
    doğrudan embedlenir (eski davranış)."""
    import torch
    embs = {}
    npz_path = bk["photo_npz"]
    if npz_path and npz_path.exists():
        d = np.load(npz_path, allow_pickle=False)
        cached = dict(zip(d["filenames"].tolist(), d["embeddings"].astype(np.float32)))
        embs = {p: cached[p] for p in photo_names if p in cached}
    missing = [p for p in photo_names if p not in embs]
    if missing:
        embedder = make_embedder(cfg, bk["model"])
        embs.update(collect_photo_embeddings(embedder, missing, cfg))
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if npz_path:  # cache'i büyüt (bir dahaki koşu embedlemesin)
            allmap = cached if npz_path.exists() else {}
            allmap.update(embs)
            names = sorted(allmap)
            np.savez(npz_path, embeddings=np.stack([allmap[n] for n in names]),
                     filenames=np.array(names))
    return embs


def collect_photo_embeddings(embedder, photo_names, cfg):
    """Etiketli fotoğrafların lineart embedding'leri (04'ün cache'inden)."""
    cache_dir = EVAL_DIR / "cache"
    photos_dir = ROOT / cfg["paths"]["photos"]
    images, kept = [], []
    fresh = 0
    photo_mod = None
    for name in photo_names:
        cp = cache_dir / (name + ".lineart.png")
        try:
            if cp.exists():
                with Image.open(cp) as im:
                    im.load()
                    images.append(im.convert("RGB"))
            else:
                if photo_mod is None:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "photo_to_lineart", ROOT / "scripts" / "02_photo_to_lineart.py")
                    photo_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(photo_mod)
                lineart = photo_mod.process_photo(photos_dir / name, cfg["image_size"])
                cache_dir.mkdir(parents=True, exist_ok=True)
                lineart.save(cp)
                images.append(lineart)
                fresh += 1
            kept.append(name)
        except Exception as e:
            print(f"  ATLANDI {name}: {e}")
    if fresh:
        print(f"({fresh} fotoğraf cache'te yoktu, pipeline'dan geçirildi)")
    embs = {}
    for i in range(0, len(images), 32):
        for n, v in zip(kept[i:i + 32], embedder.embed_batch(images[i:i + 32])):
            embs[n] = v.astype(np.float32)
    return embs


# ------------------------------------------------------------------ model/eğitim

def make_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(EMB_DIM, EMB_DIM), nn.GELU(approximate="tanh"),
        nn.Linear(EMB_DIM, PROJ_DIM),
    )


def project_np(model, mat: np.ndarray, device) -> np.ndarray:
    """(N,768) numpy -> (N,512) L2-normalize numpy (değerlendirme için)."""
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(mat), 4096):
            t = torch.from_numpy(mat[i:i + 4096]).to(device)
            y = model(t)
            y = torch.nn.functional.normalize(y, dim=1)
            outs.append(y.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def recall_at(proj_photos, targets_list, cad_proj, cad_names, ks=(1, 5, 10, 20)):
    """proj_photos: (n,512), targets_list: her satır için doğru ad kümesi."""
    pos = {n: i for i, n in enumerate(cad_names)}
    hits = {k: 0 for k in ks}
    n = 0
    max_k = max(ks)
    for q, targets in zip(proj_photos, targets_list):
        tid = [pos[t] for t in targets if t in pos]
        if not tid:
            continue
        n += 1
        scores = cad_proj @ q
        top = np.argpartition(-scores, max_k)[:max_k]
        top = top[np.argsort(-scores[top])]
        rank = next((r for r, i in enumerate(top, 1) if i in set(tid)), None)
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
    return {f"r@{k}": (hits[k] / n if n else 0.0) for k in ks}, n


def info_nce(model, left, right, right_names, device, temp=TEMPERATURE,
             extra_right=None, extra_names=None, twin_same=None):
    """Simetrik InfoNCE. Aynı CAD'e giden farklı satırlar birbirinin
    false-negative'i olmasın diye logit maskesi uygulanır.

    twin_same: (n,n) bool — batch'teki CAD çiftlerinden base-uzayda ikiz
    olanlar (katalog yakın-kopya dolu; ikizini pozitiften itmek çelişkili
    ders olur). Ad-eşitliği maskesiyle birleştirilir, köşegen korunur.

    extra_right/extra_names: zor negatifler — yalnız satır (foto->CAD)
    yönünde ek negatif kolon olarak eklenir; bir satırın pozitif CAD'iyle
    aynı ada sahip kolonlar o satır için maskelenir. Sütun (CAD->foto)
    yönündeki kayıp değişmez."""
    import torch
    import torch.nn.functional as F
    zl = F.normalize(model(left), dim=1)
    zr = F.normalize(model(right), dim=1)
    logits = zl @ zr.T / temp
    n = len(right_names)
    same = right_names[:, None] == right_names[None, :]
    if twin_same is not None:
        same = same | twin_same
    np.fill_diagonal(same, False)
    mask = torch.from_numpy(same).to(device)
    logits = logits.masked_fill(mask, float("-inf"))
    labels = torch.arange(n, device=device)
    loss_cols = F.cross_entropy(logits.T, labels)
    if extra_right is not None:
        ze = F.normalize(model(extra_right), dim=1)
        ext = zl @ ze.T / temp
        same_e = right_names[:, None] == extra_names[None, :]
        ext = ext.masked_fill(torch.from_numpy(same_e).to(device), float("-inf"))
        logits = torch.cat([logits, ext], dim=1)
    return (F.cross_entropy(logits, labels) + loss_cols) / 2


def cmd_mine(args, cfg):
    """Zor negatif madenciliği: mevcut dev modelin (projection.pt) eğitim
    fotoğraflarında karıştırdığı YANLIŞ CAD'leri çıkarır; train --hard-negatives
    bunları satır-özel negatif olarak kayba ekler.

    İkiz koruması: fotoğrafın pozitiflerine base-uzayda >= --hn-twin-sim
    benzeyen adaylar (muhtemel etiketlenmemiş kopya/varyant) negatif SAYILMAZ —
    gerçekte doğru olabilecek bir çizimi cezalandırmamak için.
    Çıktı: data/train_cache/hard_negatives.json. Model ya da etiketler
    değiştikçe yeniden koşulmalı (eski maden eski modelin hatalarıdır)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bk = BACKBONES[args.backbone]
    common, _ = aligned_names(cfg, (bk["clean"], bk["hed"]))
    excluded = load_excluded_trivial(cfg) if cfg.get("exclude_trivial") else set()
    if excluded:
        common = [n for n in common if n not in excluded]
    clean_base = load_side(cfg, bk["clean"], common)
    base_n = clean_base / np.clip(
        np.linalg.norm(clean_base, axis=1, keepdims=True), 1e-9, None)

    eslesme, kaynak = load_labels_clean(args.allow_dirty_labels)
    cad_set = set(common)
    usable = {p: [Path(c).name for c in cads if Path(c).name in cad_set]
              for p, cads in eslesme.items()}
    usable = {p: c for p, c in usable.items() if c}
    split_path = EVAL_DIR / "projection_split.json"
    if not split_path.exists():
        print("projection_split.json yok — madencilikten önce normal 'train' gerekli.")
        sys.exit(1)
    with open(split_path, "r", encoding="utf-8") as f:
        prev = json.load(f)
    train_photos, _, _ = frozen_split(sorted(usable), prev)

    ckpt = torch.load(MODELS_DIR / "projection.pt", map_location="cpu")
    model = make_model().to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Maden modeli: projection.pt (epoch {ckpt.get('epoch', '?')})")
    cad_proj = project_np(model, clean_base, device)

    photo_embs = collect_photo_embeddings_bk(bk, train_photos, cfg)

    name_pos = {n: i for i, n in enumerate(common)}
    kept_photos = [p for p in train_photos if p in photo_embs]
    q_proj = project_np(model, np.stack([photo_embs[p] for p in kept_photos]), device)

    out, counts, twin_dropped = {}, [], 0
    for p, q in zip(kept_photos, q_proj):
        scores = cad_proj @ q
        pool = min(args.hn_pool, len(scores) - 1)
        top = np.argpartition(-scores, pool)[:pool]
        top = top[np.argsort(-scores[top])]
        pos_set = {name_pos[c] for c in usable[p]}
        pos_vecs = base_n[sorted(pos_set)]
        negs = []
        for i in top:
            if int(i) in pos_set:
                continue
            if float((base_n[i] @ pos_vecs.T).max()) >= args.hn_twin_sim:
                twin_dropped += 1
                continue
            negs.append(common[int(i)])
            if len(negs) >= args.hn_per_photo:
                break
        out[p] = negs
        counts.append(len(negs))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "hard_negatives.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Zor negatifler -> data/train_cache/hard_negatives.json: "
          f"{len(out)} foto, foto başına ort. {np.mean(counts):.1f} negatif "
          f"(ikiz korumasıyla düşülen aday: {twin_dropped})")


def cmd_prepare(args, cfg):
    """Augmentation'lı embedding cache'leri (sentetik çiftlerin ek kopyaları)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bk = BACKBONES[args.backbone]
    common, _ = aligned_names(cfg, (bk["clean"], bk["hed"]))
    if args.limit:
        common = common[:args.limit]
    print(f"{len(common)} ortak CAD | augment kopya/taraf: {args.aug_per_side} | omurga: {args.backbone}")
    embedder = make_embedder(cfg, bk["model"])
    sides = [("clean", ROOT / cfg["paths"]["cad_clean"]),
             ("hed", ROOT / "data" / "variants" / "cad_hedthick5")]
    for a in range(args.aug_per_side):
        for tag, src in sides:
            out = CACHE_DIR / f"{bk['aug_prefix']}{tag}_aug{a + 1}.npz"
            if out.exists():
                print(f"  {out.name} zaten var, atlanıyor")
                continue
            embed_folder(embedder, src, common, out, augment=True,
                         batch_size=cfg["batch_size"], seed=SEED + a * 7 + (tag == "hed"))


def load_aug_mats(tag: str, common, prefix=""):
    """data/train_cache'teki augment matrisleri, 'common' ad sırasına hizalanır.
    Cache tam listeyle üretilmiş olabilir (prepare dışlama uygulamaz); common
    onun alt kümesiyse satırlar seçilerek kullanılır — exclude_trivial açılıp
    kapansa da cache geçerli kalır."""
    mats = []
    for f in sorted(CACHE_DIR.glob(f"{prefix}{tag}_aug*.npz")):
        d = np.load(f, allow_pickle=False)
        names = [n for n in d["filenames"].tolist()]
        if names == common:
            mats.append(d["embeddings"].astype(np.float32))
        else:
            pos = {n: i for i, n in enumerate(names)}
            missing = [n for n in common if n not in pos]
            if missing:
                print(f"  UYARI: {f.name} {len(missing)} adı içermiyor, atlanıyor "
                      f"(yeniden üretim: dosyayı silip 'prepare')")
                continue
            emb = d["embeddings"].astype(np.float32)
            mats.append(emb[[pos[n] for n in common]])
        print(f"  augment cache: {f.name} ({len(common)}/{len(names)} satır)")
    return mats


def cmd_train(args, cfg):
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- sentetik taraf
    bk = BACKBONES[args.backbone]
    common, _ = aligned_names(cfg, (bk["clean"], bk["hed"]))
    excluded = load_excluded_trivial(cfg) if cfg.get("exclude_trivial") else set()
    if excluded:
        before = len(common)
        common = [n for n in common if n not in excluded]
        print(f"exclude_trivial: sentetik kümeden {before - len(common)} CAD düşüldü")
    clean_mats = [load_side(cfg, bk["clean"], common)]
    hed_mats = [load_side(cfg, bk["hed"], common)]
    if not args.no_aug:
        clean_mats += load_aug_mats("clean", common, bk["aug_prefix"])
        hed_mats += load_aug_mats("hed", common, bk["aug_prefix"])
    if len(clean_mats) == 1 and not args.no_aug:
        print("Not: augment cache yok ('prepare' koşulmadı) — sadece base embedding'lerle.")
    n_syn = len(common)
    print(f"Sentetik çift: {n_syn} (clean x{len(clean_mats)}, hed x{len(hed_mats)} varyasyon)")

    # --- gerçek taraf: aile bazlı eğitim/val/test ayrımı
    eslesme, kaynak = load_labels_clean(args.allow_dirty_labels)
    cad_set = set(common)
    usable = {p: [Path(c).name for c in cads if Path(c).name in cad_set]
              for p, cads in eslesme.items()}
    usable = {p: c for p, c in usable.items() if c}
    photo_tags = {p: {kaynak.get(p, {}).get(c, "manual") for c in cads}
                  for p, cads in usable.items()}
    split_path = EVAL_DIR / "projection_split.json"
    if args.final:
        # ÜRETİM modeli: tüm etiketli veri eğitimde, ölçüm yok. Ayrım dosyalarına
        # ve projection.pt / index/variants/projected'a DOKUNULMAZ — dürüst
        # kıyaslamalar donmuş ayrımlı normal eğitimle yapılmaya devam eder.
        train_photos, val_photos, test_photos = sorted(usable), [], []
        print(f"FINAL eğitim: TÜM {len(train_photos)} etiketli foto eğitimde "
              f"(val/test yok, epoch sabit {args.epochs})")
    elif split_path.exists() and not args.resplit:
        with open(split_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        train_photos, val_photos, test_photos = frozen_split(sorted(usable), prev)
    else:
        train_photos, val_photos, test_photos = family_split(sorted(usable), photo_tags)
    if not args.final:
        with open(EVAL_DIR / "projection_split.json", "w", encoding="utf-8") as f:
            json.dump({"train": train_photos, "val": val_photos, "test": test_photos},
                      f, ensure_ascii=False, indent=1)
        for name, lst in (("projection_train_photos.json", train_photos),
                          ("projection_val_photos.json", val_photos),
                          ("projection_test_photos.json", test_photos)):
            with open(EVAL_DIR / name, "w", encoding="utf-8") as f:
                json.dump(lst, f, ensure_ascii=False, indent=1)

    photo_embs = collect_photo_embeddings_bk(
        bk, train_photos + val_photos + test_photos, cfg)

    name_pos = {n: i for i, n in enumerate(common)}
    hn_map = {}
    if args.hard_negatives:
        hn_path = CACHE_DIR / "hard_negatives.json"
        if not hn_path.exists():
            print("hard_negatives.json yok — önce 'mine' aşamasını koşun.")
            sys.exit(1)
        with open(hn_path, "r", encoding="utf-8") as f:
            raw_hn = json.load(f)
        hn_map = {p: np.array([name_pos[c] for c in cads if c in name_pos],
                              dtype=np.int64)
                  for p, cads in raw_hn.items()}
        covered = sum(1 for p in train_photos if len(hn_map.get(p, ())) > 0)
        print(f"Zor negatifler: {len(hn_map)} foto madenli, eğitimdekilerin "
              f"{covered}/{len(train_photos)}'i kapsanıyor "
              f"(çift başına {args.hn_per_real} negatif eklenecek)")
    real_pairs = [(photo_embs[p], name_pos[c], c, hn_map.get(p))
                  for p in train_photos if p in photo_embs
                  for c in usable[p]]
    print(f"Gerçek eğitim çifti: {len(real_pairs)} "
          f"({sum(1 for p in train_photos if p in photo_embs)} fotoğraftan)")

    def subset(photos):
        kept = [p for p in photos if p in photo_embs]
        if not kept:  # --final modunda val/test boş
            return np.zeros((0, EMB_DIM), dtype=np.float32), []
        return np.stack([photo_embs[p] for p in kept]), [set(usable[p]) for p in kept]

    train_q, train_targets = subset(train_photos)
    val_q, val_targets = subset(val_photos)
    test_q, test_targets = subset(test_photos)  # eğitim boyunca HİÇ ölçülmez

    # --- model + optimizasyon
    model = make_model().to(device)
    steps_per_epoch = max(1, n_syn // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)
    rng = np.random.default_rng(SEED)
    clean_base = clean_mats[0]

    best = {"r@10": -1.0, "epoch": -1, "state": None}
    patience_left = args.patience
    print(f"Cihaz: {device} | epoch: {args.epochs} | batch: {args.batch_size} "
          f"(+{args.real_per_batch} gerçek) | lr: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n_syn)
        losses = []
        for s in range(steps_per_epoch):
            idx = order[s * args.batch_size:(s + 1) * args.batch_size]
            l_mat = hed_mats[rng.integers(len(hed_mats))][idx]
            r_mat = clean_mats[rng.integers(len(clean_mats))][idx]
            names = [common[i] for i in idx]
            extra_r = extra_names = None
            if real_pairs and args.real_per_batch:
                picks = rng.integers(len(real_pairs), size=args.real_per_batch)
                l_mat = np.concatenate([l_mat, np.stack([real_pairs[i][0] for i in picks])])
                r_mat = np.concatenate([r_mat, clean_base[[real_pairs[i][1] for i in picks]]])
                names += [real_pairs[i][2] for i in picks]
                if args.hard_negatives:
                    eidx = []
                    for i in picks:
                        lst = real_pairs[i][3]
                        if lst is None or not len(lst):
                            continue
                        take = min(args.hn_per_real, len(lst))
                        eidx.extend(rng.choice(lst, size=take, replace=False).tolist())
                    if eidx:
                        extra_r = torch.from_numpy(clean_base[eidx]).to(device)
                        extra_names = np.array([common[i] for i in eidx])
            twin_same = None
            if args.twin_mask_sim:
                bv = clean_base[[name_pos[nm] for nm in names]]
                twin_same = (bv @ bv.T) >= args.twin_mask_sim
            loss = info_nce(model,
                            torch.from_numpy(l_mat).to(device),
                            torch.from_numpy(r_mat).to(device),
                            np.array(names), device,
                            extra_right=extra_r, extra_names=extra_names,
                            twin_same=twin_same)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.detach().item())

        if args.final:
            # ölçülecek ayrılmış veri yok: her epoch son model "en iyi" sayılır
            best = {"r@10": 0.0, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            print(f"epoch {epoch:2d} | loss {np.mean(losses):.4f}")
            continue

        # validation r@10 (erken durdurma metriği; test'e eğitim boyunca BAKILMAZ)
        cad_proj = project_np(model, clean_base, device)
        val_r, _ = recall_at(project_np(model, val_q, device), val_targets,
                             cad_proj, common)
        print(f"epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
              f"val r@1 {val_r['r@1']:.1%} r@10 {val_r['r@10']:.1%} "
              f"r@20 {val_r['r@20']:.1%}")
        if val_r["r@10"] > best["r@10"]:
            best = {"r@10": val_r["r@10"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Erken durdurma: {args.patience} epoch'tur iyileşme yok "
                      f"(en iyi epoch {best['epoch']}).")
                break

    model.load_state_dict(best["state"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.out_variant:
        model_path = MODELS_DIR / f"{args.out_variant}.pt"
    else:
        model_path = MODELS_DIR / ("projection_final.pt" if args.final else "projection.pt")
    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "dims": [EMB_DIM, PROJ_DIM], "model": bk["model"],
                "backbone": args.backbone},
               model_path)
    print(f"En iyi model (epoch {best['epoch']}) -> {model_path}")

    cad_proj = project_np(model, clean_base, device)
    train_r, n_tr = recall_at(project_np(model, train_q, device), train_targets, cad_proj, common)
    if args.final:
        print(f"\n=== FINAL MODEL (epoch {best['epoch']}, tüm etiketli veri) ===")
        print(f"eğitim ({n_tr} foto): " + "  ".join(f"{k} {v:6.1%}" for k, v in train_r.items()))
        print("(ayrılmış veri kalmadığı için dürüst ölçüm YOK; eğitim sayısı ezber içerir. "
              "Kıyas için donmuş ayrımlı normal eğitimin test sonucu esas alınır.)")
        cmd_build(args, cfg)
        return

    # --- final rapor: test burada İLK VE TEK KEZ ölçülür
    val_r, n_va = recall_at(project_np(model, val_q, device), val_targets, cad_proj, common)
    test_r, n_te = recall_at(project_np(model, test_q, device), test_targets, cad_proj, common)
    print(f"\n=== PROJEKSİYON SONUÇ (epoch {best['epoch']}) ===")
    print(f"eğitim ({n_tr} foto): " + "  ".join(f"{k} {v:6.1%}" for k, v in train_r.items()))
    print(f"val    ({n_va} foto): " + "  ".join(f"{k} {v:6.1%}" for k, v in val_r.items()))
    print(f"test   ({n_te} foto): " + "  ".join(f"{k} {v:6.1%}" for k, v in test_r.items()))
    print("(eğitim >> val/test ise overfitting var demektir; test tek kez ölçüldü)")
    with open(EVAL_DIR / "projection_report.json", "w", encoding="utf-8") as f:
        json.dump({"epoch": best["epoch"], "egitim": train_r, "val": val_r,
                   "test": test_r, "n_egitim": n_tr, "n_val": n_va, "n_test": n_te,
                   "sentetik": n_syn, "gercek_cift": len(real_pairs)},
                  f, ensure_ascii=False, indent=2)

    cmd_build(args, cfg)


def cmd_build(args, cfg):
    """En iyi modelle index/variants/projected kurulumu (04 --variant projected).
    --final ile: projection_final.pt -> index/variants/projected_final (üretim)."""
    import torch
    import faiss

    if args.out_variant:
        variant, model_file = args.out_variant, f"{args.out_variant}.pt"
    else:
        variant = "projected_final" if args.final else "projected"
        model_file = "projection_final.pt" if args.final else "projection.pt"
    ckpt = torch.load(MODELS_DIR / model_file, map_location="cpu")
    model = make_model()
    model.load_state_dict(ckpt["state_dict"])
    bk = BACKBONES[ckpt.get("backbone", args.backbone)]
    if ckpt.get("model") and ckpt["model"] != bk["model"]:
        print(f"UYARI: checkpoint modeli ({ckpt['model']}) ile omurga ({bk['model']}) uyumsuz!")

    common, _ = aligned_names(cfg, (bk["clean"], bk["hed"]))
    clean = load_side(cfg, bk["clean"], common)
    proj = project_np(model, clean, "cpu")

    out_dir = ROOT / cfg["paths"]["index_dir"] / "variants" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cad_embeddings.npy", proj)
    with open(out_dir / "cad_filenames.json", "w", encoding="utf-8") as f:
        json.dump(common, f, ensure_ascii=False)
    index = faiss.IndexFlatIP(proj.shape[1])
    index.add(proj)
    (out_dir / "faiss.index").write_bytes(faiss.serialize_index(index).tobytes())
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": ckpt["model"], "kaynak": "index(base) + projection",
                   "varyant": variant, "epoch": ckpt["epoch"]},
                  f, ensure_ascii=False, indent=2)
    # search.py sorgu tarafında da aynı projeksiyonu uygulasın diye numpy ağırlıklar
    sd = ckpt["state_dict"]
    np.savez(out_dir / "projection.npz",
             W1=sd["0.weight"].numpy().T, b1=sd["0.bias"].numpy(),
             W2=sd["2.weight"].numpy().T, b2=sd["2.bias"].numpy())
    print(f"Projeksiyonlu indeks -> {out_dir} ({proj.shape})")
    print("Ölçüm: python scripts/04_evaluate.py --variant projected "
          "--photos-file data/eval/projection_test_photos.json")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 üstüne projeksiyon eğitimi")
    parser.add_argument("stage", choices=["prepare", "train", "build", "all", "mine"])
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="dinov2",
                        help="Embedding omurgası (CAD varyant çifti + foto modeli)")
    parser.add_argument("--out-variant", type=str, default=None,
                        help="Çıktı varyant/checkpoint adını ez (füzyon için ikinci "
                             "final: ör. projected_final_dinov2 — projected_final'ı ezmez)")
    parser.add_argument("--aug-per-side", type=int, default=1,
                        help="prepare: taraf başına augment kopya sayısı")
    parser.add_argument("--limit", type=int, default=None, help="prepare: ilk N CAD (deneme)")
    parser.add_argument("--no-aug", action="store_true",
                        help="train: augment cache'lerini kullanma")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--real-per-batch", type=int, default=64,
                        help="Batch'e eklenen gerçek çift sayısı")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5,
                        help="Erken durdurma sabrı (epoch)")
    parser.add_argument("--final", action="store_true",
                        help="ÜRETİM eğitimi: tüm etiketli veri eğitimde, val/test ve "
                             "erken durdurma yok (--epochs sabit); çıktı "
                             "projection_final.pt + index/variants/projected_final. "
                             "Ayrım dosyalarına ve projected varyantına dokunmaz.")
    parser.add_argument("--resplit", action="store_true",
                        help="Önceki projection_split.json'u yok sayıp ayrımı sıfırdan kur "
                             "(test karşılaştırılabilirliği BOZULUR)")
    parser.add_argument("--allow-dirty-labels", action="store_true",
                        help="labels_clean.json yoksa ham labels.json ile eğit")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="train: mine çıktısındaki zor negatifleri kayba ekle")
    parser.add_argument("--hn-per-real", type=int, default=4,
                        help="train: gerçek çift başına eklenen zor negatif")
    parser.add_argument("--hn-per-photo", type=int, default=10,
                        help="mine: foto başına saklanan negatif sayısı")
    parser.add_argument("--hn-pool", type=int, default=100,
                        help="mine: negatif seçilen top-N aday havuzu")
    parser.add_argument("--hn-twin-sim", type=float, default=0.95,
                        help="mine: pozitife bu base-benzerlik üstündeki aday "
                             "negatif sayılmaz (etiketlenmemiş ikiz koruması)")
    parser.add_argument("--twin-mask-sim", type=float, default=0.0,
                        help="train: batch içi negatiflerden, pozitife bu "
                             "base-benzerlik üstünde olanlar maskelenir "
                             "(0 = kapalı; katalog ikizleri false-negative olmasın)")
    args = parser.parse_args()

    cfg = load_config()
    if args.stage == "mine":
        cmd_mine(args, cfg)
        return
    if args.stage in ("prepare", "all"):
        cmd_prepare(args, cfg)
    if args.stage in ("train", "all"):
        cmd_train(args, cfg)   # sonunda build'i de çağırır
    elif args.stage == "build":
        cmd_build(args, cfg)


if __name__ == "__main__":
    main()
