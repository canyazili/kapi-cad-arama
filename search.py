# -*- coding: utf-8 -*-
"""
search.py — Arama çekirdeği.

Tüm arama mantığı buradadır; Streamlit arayüzü (app/main.py) ve
değerlendirme scripti (scripts/04_evaluate.py) bu modülü import eder.

Akış: fotoğraf -> kapı-crop + HED lineart (02 scriptindeki process_photo)
      -> DINOv2 embedding -> FAISS IndexFlatIP araması.

DINOv2 embedder'ı scripts/03_build_index.py de buradan import eder;
indeksleme ve arama aynı modeli/aynı ön işlemeyi kullanır.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def _find_root() -> Path:
    """Proje kökü. Normalde bu dosyanın klasörü; PyInstaller exe'sinde __file__
    bundle içini gösterdiğinden KAPI_ROOT ortam değişkenine ya da exe'nin
    bulunduğu yerden yukarı doğru configs/config.yaml aramasına düşülür."""
    env = os.environ.get("KAPI_ROOT")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        d = Path(sys.executable).resolve().parent
        for c in (d, *d.parents):
            if (c / "configs" / "config.yaml").exists():
                return c
    return Path(__file__).resolve().parent


ROOT = _find_root()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_photo_module():
    """scripts/02_photo_to_lineart.py sayıyla başladığı için importlib ile yüklenir."""
    path = ROOT / "scripts" / "02_photo_to_lineart.py"
    spec = importlib.util.spec_from_file_location("photo_to_lineart", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DinoEmbedder:
    """DINOv2 (torch.hub) ile L2-normalize embedding üretir. GPU varsa kullanır."""

    def __init__(self, cfg=None, model_name=None):
        import torch
        cfg = cfg or load_config()
        self.model_name = model_name or cfg["model"]["name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load(cfg["model"]["hub_repo"], self.model_name)
        self.model.eval().to(self.device)
        self.image_size = cfg["image_size"]

    def _to_tensor(self, pil_img):
        import torch
        img = pil_img.convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))

    def embed_batch(self, pil_images):
        """PIL görsel listesi -> (N, D) float32, satırlar L2-normalize."""
        import torch
        batch = torch.stack([self._to_tensor(im) for im in pil_images]).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        feats = feats.cpu().numpy().astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        return feats

    def embed(self, pil_img):
        return self.embed_batch([pil_img])[0]


class Dinov3Embedder:
    """DINOv3 (HuggingFace transformers) ile L2-normalize embedding üretir.
    DinoEmbedder ile aynı arayüz (embed/embed_batch). Ağırlıklar HF cache'inden
    yüklenir (taşınabilir pakette modeller/hf'e yönlendirilir, HF_HUB_OFFLINE=1)."""

    HF_IDS = {"dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m"}
    IMAGE_SIZE = 512  # patch16 katı (dinov2'deki 518'in dengi)

    def __init__(self, cfg=None, model_name="dinov3_vitb16"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.model_name = model_name
        hf_id = self.HF_IDS[model_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.processor.size = {"height": self.IMAGE_SIZE, "width": self.IMAGE_SIZE}
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(hf_id, dtype=dtype).eval().to(self.device)

    def embed_batch(self, pil_images):
        import torch
        images = [im.convert("RGB") for im in pil_images]
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs = {k: (v.half() if v.dtype == torch.float32 else v)
                          for k, v in inputs.items()}
            feats = self.model(**inputs).pooler_output
        feats = feats.float().cpu().numpy().astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        return feats

    def embed(self, pil_img):
        return self.embed_batch([pil_img])[0]


def make_embedder(cfg=None, model_name=None):
    """Model adına göre doğru embedder'ı döner (dinov3_* -> HF, diğerleri torch.hub)."""
    cfg = cfg or load_config()
    name = model_name or cfg["model"]["name"]
    if name.startswith("dinov3"):
        return Dinov3Embedder(cfg, model_name=name)
    return DinoEmbedder(cfg, model_name=name)


def resolve_index_dir(cfg, variant=None) -> Path:
    """Varyant adına göre indeks klasörünü döner ('base' -> index/)."""
    variant = variant or cfg.get("active_index", "base")
    base = ROOT / cfg["paths"]["index_dir"]
    return base if variant in (None, "", "base") else base / "variants" / variant


def load_excluded_trivial(cfg=None) -> set:
    """index/excluded_trivial.json'daki (08_filter_trivial onayı) dosya adları."""
    cfg = cfg or load_config()
    path = ROOT / cfg["paths"]["index_dir"] / "excluded_trivial.json"
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


class NpzProjection:
    """07_train_projection çıktısı: embedding üstüne 2 katmanlı MLP (numpy).
    İndeks klasöründeki projection.npz'den yüklenir; torch gerektirmez.
    GELU tanh yaklaşımı kullanılır — eğitimdekiyle (approximate='tanh') birebir."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(D,) veya (N, D) alır; L2-normalize edilmiş projeksiyon döner."""
        single = x.ndim == 1
        h = self._gelu(np.atleast_2d(x) @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-8
        return y[0] if single else y


class SearchEngine:
    """FAISS indeksini ve modelleri bir kez yükleyip tekrar tekrar arama yapar.

    variant: indeks varyantı adı (None -> config'teki active_index).
    Varyantın meta.json'ında model adı varsa sorgu embedding'i de o modeli kullanır.
    exclude_trivial: None -> config'teki exclude_trivial anahtarı; True/False ile ezilebilir.
    İndeks klasöründe projection.npz varsa (07_train_projection çıktısı) sorgu
    embedding'i aramadan önce projeksiyondan geçirilir.
    """

    def __init__(self, variant=None, exclude_trivial=None):
        import faiss
        self.cfg = load_config()
        self.variant = variant or self.cfg.get("active_index", "base") or "base"
        if exclude_trivial is None:
            exclude_trivial = bool(self.cfg.get("exclude_trivial", False))
        self.excluded = load_excluded_trivial(self.cfg) if exclude_trivial else set()
        if self.excluded:
            print(f"exclude_trivial: {len(self.excluded)} CAD arama dışı")
        self.projection = None
        self.index = None
        if self.variant == "fusion":
            # Füzyon modu: birden çok varyantın skorları ağırlıklı toplanır
            # (deneylerde tekil her indeksten daha iyi çıktı).
            self._init_fusion()
            self.photo_mod = _load_photo_module()
            return
        index_dir = resolve_index_dir(self.cfg, self.variant)
        index_path = index_dir / "faiss.index"
        names_path = index_dir / "cad_filenames.json"
        if not index_path.exists() or not names_path.exists():
            raise FileNotFoundError(
                f"İndeks bulunamadı ({index_path}). Önce scripts/03_build_index.py çalıştırın.")
        # faiss.read_index Windows'ta Türkçe karakterli yollarda çalışmıyor;
        # dosya Python ile okunup deserialize ediliyor (03 scripti de böyle yazar).
        buf = np.fromfile(index_path, dtype=np.uint8)
        self.index = faiss.deserialize_index(buf)
        with open(names_path, "r", encoding="utf-8") as f:
            self.filenames = json.load(f)

        model_name = None
        meta_path = index_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                model_name = json.load(f).get("model")
        proj_path = index_dir / "projection.npz"
        if proj_path.exists():
            self.projection = NpzProjection(proj_path)
            print(f"Projeksiyon katmanı yüklendi: {proj_path.relative_to(ROOT)}")
        self.embedder = make_embedder(self.cfg, model_name=model_name)
        self.photo_mod = _load_photo_module()

    def _init_fusion(self):
        """config'teki fusion.variants indekslerini yükler. Varyantlar FARKLI
        omurgalar kullanabilir (ör. projected_final [dinov3] +
        projected_final_dinov2): her varyantın kendi embedder'ı ve
        projection.npz'si sorguya ayrı uygulanır, skorlar ağırlıkla toplanır.
        Dosya adları ortak kesişime hizalanır."""
        fus = self.cfg.get("fusion") or {}
        variants = fus.get("variants", ["base"])
        weights = [float(w) for w in fus.get("weights", [1.0] * len(variants))]
        sides, names_ref = [], None
        for v, w in zip(variants, weights):
            d = resolve_index_dir(self.cfg, v)
            emb_path = d / "cad_embeddings.npy"
            if not emb_path.exists():
                raise FileNotFoundError(f"Füzyon varyantı eksik: {emb_path}")
            emb = np.load(emb_path).astype(np.float32)
            with open(d / "cad_filenames.json", "r", encoding="utf-8") as f:
                names = json.load(f)
            model = self.cfg["model"]["name"]
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    model = json.load(f).get("model", model)
            proj_path = d / "projection.npz"
            proj = NpzProjection(proj_path) if proj_path.exists() else None
            sides.append({"w": w, "emb": emb, "names": names,
                          "model": model, "proj": proj})
            names_ref = (list(names) if names_ref is None
                         else [n for n in names_ref if n in set(names)])
        self.filenames = names_ref
        for s in sides:  # ortak ada hizala
            pos = {n: i for i, n in enumerate(s["names"])}
            s["emb"] = s["emb"][[pos[n] for n in names_ref]]
            del s["names"]
        # aynı omurga iki varyantta kullanılıyorsa embedder'ı bir kez yükle
        self.fusion_embedders = {}
        for s in sides:
            if s["model"] not in self.fusion_embedders:
                self.fusion_embedders[s["model"]] = make_embedder(
                    self.cfg, model_name=s["model"])
            print(f"füzyon tarafı: {s['model']} (w={s['w']}"
                  f"{', projeksiyonlu' if s['proj'] else ''})")
        self.fusion_sides = sides
        self.embedder = next(iter(self.fusion_embedders.values()))

    def prepare_query(self, photo):
        """Fotoğrafı arama girdisine çevirir; (lineart, cropped, cleaned) döner
        (arayüz debug için; metin silme kapalıysa cleaned None'dır)."""
        return self.photo_mod.process_photo(photo, self.cfg["image_size"], return_steps=True)

    def search_prepared(self, lineart, k=None):
        """Hazır lineart görseliyle arama; [(dosya_adı, skor)] döner.
        excluded_trivial listesindeki CAD'ler sonuçlardan düşülür."""
        k = k or self.cfg["top_k"]
        if self.index is None:  # füzyon modu: taraf başına sorgu + ağırlıklı toplam
            q_cache, scores = {}, None
            for s in self.fusion_sides:
                if s["model"] not in q_cache:
                    q_cache[s["model"]] = self.fusion_embedders[s["model"]].embed(lineart)
                q = q_cache[s["model"]]
                if s["proj"] is not None:
                    q = s["proj"](q).astype(np.float32)
                part = s["w"] * (s["emb"] @ q)
                scores = part if scores is None else scores + part
            top = np.argsort(-scores)[:k + len(self.excluded)]
            return [(self.filenames[i], float(scores[i])) for i in top
                    if self.filenames[i] not in self.excluded][:k]
        q = self.embedder.embed(lineart)
        if self.projection is not None:
            q = self.projection(q).astype(np.float32)
        # dışlananlar sonuçtan düşülünce k'nın altına inmemek için fazladan iste
        kk = min(k + len(self.excluded), self.index.ntotal)
        s, ids = self.index.search(q.reshape(1, -1), kk)
        return [(self.filenames[i], float(v))
                for i, v in zip(ids[0], s[0])
                if i >= 0 and self.filenames[i] not in self.excluded][:k]

    def search(self, photo, k=None):
        """photo: dosya yolu veya PIL.Image. [(dosya_adı, skor)] döner."""
        lineart, _, _ = self.prepare_query(photo)
        return self.search_prepared(lineart, k)


_engines = {}


def get_engine(variant=None, exclude_trivial=None) -> SearchEngine:
    """Varyant başına tek SearchEngine örneği (modeller bir kez yüklensin)."""
    key = (variant or "config", exclude_trivial)
    if key not in _engines:
        _engines[key] = SearchEngine(variant=variant, exclude_trivial=exclude_trivial)
    return _engines[key]


def search(photo_path, k=20):
    """Basit API: fotoğraf yolu ver, [(cad_dosya_adı, benzerlik_skoru)] al."""
    return get_engine().search(photo_path, k)
