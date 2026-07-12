# -*- coding: utf-8 -*-
"""
03_build_index.py — CAD embedding'leri ve FAISS indeksi.

data/cad_clean içindeki normalize CAD görsellerini DINOv2 ile batch batch
embedler (GPU varsa GPU'da), L2-normalize eder ve FAISS IndexFlatIP kurar.

Çıktılar (index/):
  - cad_embeddings.npy   (N, D) float32
  - cad_filenames.json   embedding satırlarıyla AYNI sıradaki dosya adları
  - faiss.index

Sıra garantisi: dosya listesi deterministik sıralanır; embedding matrisi ve
dosya adı listesi aynı döngüde, aynı sırayla üretilir (sıra kayması = yanlış sonuç).

Checkpoint: her 'checkpoint_every' batch'te ara durum index/checkpoint.npz'ye
yazılır; script yarıda kesilirse kaldığı yerden devam eder.

Kullanım:
  python scripts/03_build_index.py             # tüm veri
  python scripts/03_build_index.py --limit 100 # rastgele 100 dosyayla deneme
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search import DinoEmbedder, resolve_index_dir  # noqa: E402  (indeksleme ve arama aynı embedder'ı kullansın)


def load_config():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path: Path, filenames: list[str]):
    """Checkpoint varsa ve mevcut dosya listesinin öneki ile tutarlıysa devam noktasını döner."""
    if not ckpt_path.exists():
        return None, 0
    try:
        data = np.load(ckpt_path, allow_pickle=False)
        done_names = [n for n in data["filenames"].tolist()]
        n = len(done_names)
        if done_names == filenames[:n]:
            print(f"Checkpoint bulundu: {n} görsel hazır, kaldığı yerden devam ediliyor.")
            return data["embeddings"], n
        print("UYARI: checkpoint dosya listesiyle uyuşmuyor (veri değişmiş), baştan başlanıyor.")
    except Exception as e:
        print(f"UYARI: checkpoint okunamadı ({e}), baştan başlanıyor.")
    return None, 0


def main():
    parser = argparse.ArgumentParser(description="CAD embedding + FAISS indeksi")
    parser.add_argument("--limit", type=int, default=None, help="Rastgele N dosyayla deneme")
    parser.add_argument("--src", type=str, default=None,
                        help="Kaynak görsel klasörü (varsayılan: config'teki cad_clean)")
    parser.add_argument("--variant", type=str, default="base",
                        help="İndeks varyant adı: 'base' -> index/, diğerleri -> index/variants/<ad>/")
    parser.add_argument("--model", type=str, default=None,
                        help="DINOv2 model adı (varsayılan: config'teki model, ör. dinov2_vitl14)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch boyutu (vitl14 için 6GB GPU'da 12 önerilir)")
    args = parser.parse_args()

    cfg = load_config()
    src_dir = (ROOT / args.src) if args.src else (ROOT / cfg["paths"]["cad_clean"])
    index_dir = resolve_index_dir(cfg, args.variant)
    index_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model or cfg["model"]["name"]
    ckpt_path = index_dir / "checkpoint.npz"
    batch_size = args.batch_size or cfg["batch_size"]
    ckpt_every = cfg.get("checkpoint_every", 20)

    files = sorted(src_dir.glob("*.png"))  # deterministik sıra: sıra garantisinin temeli
    if not files:
        print(f"UYARI: {src_dir} boş. Önce scripts/01_clean_cad.py çalıştırın.", file=sys.stderr)
        sys.exit(1)

    if cfg.get("exclude_trivial"):
        from search import load_excluded_trivial
        excluded = load_excluded_trivial(cfg)
        if excluded:
            before = len(files)
            files = [f for f in files if f.name not in excluded]
            print(f"exclude_trivial: {before - len(files)} parça profili indeks dışı bırakıldı")

    if args.limit is not None:
        random.seed(42)
        files = sorted(random.sample(files, min(args.limit, len(files))))
        print(f"--limit modu: {len(files)} dosya")

    filenames = [f.name for f in files]

    prev_emb, start = load_checkpoint(ckpt_path, filenames)
    chunks = [prev_emb] if prev_emb is not None else []

    embedder = DinoEmbedder(cfg, model_name=model_name)
    print(f"Varyant: {args.variant} | Model: {model_name} | cihaz: {embedder.device} | "
          f"kaynak: {src_dir.name} | {len(files)} görsel")

    pbar = tqdm(total=len(files), initial=start, desc="Embedding", unit="img")
    batches_since_ckpt = 0
    for i in range(start, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        images = []
        for f in batch_files:
            with Image.open(f) as im:
                im.load()
                images.append(im.convert("RGB"))
        chunks.append(embedder.embed_batch(images))
        pbar.update(len(batch_files))

        batches_since_ckpt += 1
        if batches_since_ckpt >= ckpt_every:
            emb = np.concatenate(chunks, axis=0)
            chunks = [emb]
            np.savez(ckpt_path, embeddings=emb,
                     filenames=np.array(filenames[:len(emb)]))
            batches_since_ckpt = 0
    pbar.close()

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    assert len(embeddings) == len(filenames), "embedding/dosya adı sayısı uyuşmuyor!"

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])  # satırlar L2-normalize -> IP = cosine
    index.add(embeddings)

    np.save(index_dir / "cad_embeddings.npy", embeddings)
    with open(index_dir / "cad_filenames.json", "w", encoding="utf-8") as f:
        json.dump(filenames, f, ensure_ascii=False)
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "kaynak": str(src_dir.relative_to(ROOT)),
                   "varyant": args.variant}, f, ensure_ascii=False, indent=2)
    # faiss.write_index Windows'ta Türkçe karakterli yollarda (ör. "kapı") patlıyor;
    # bu yüzden indeksi serialize edip Python dosya G/Ç'siyle yazıyoruz.
    (index_dir / "faiss.index").write_bytes(faiss.serialize_index(index).tobytes())
    if ckpt_path.exists():
        ckpt_path.unlink()  # iş bitti, checkpoint'e gerek kalmadı

    print(f"Bitti: {len(filenames)} embedding ({embeddings.shape[1]} boyut) -> {index_dir}")


if __name__ == "__main__":
    main()
