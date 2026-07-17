# -*- coding: utf-8 -*-
"""DINOv3 tarama deneyi — adım 2: adil kıyas eğitimi.
Aynı tarif (augmentsız, ikiz maskesiz, aynı hiperparametreler, DONMUŞ ayrım)
ile iki projeksiyon eğitilir: dinov2 (mevcut) ve dinov3. Test (445 foto)
r@K yan yana raporlanır. Üretime DOKUNMAZ; model diske yazılmaz (sadece rapor).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "data" / "train_cache" / "dinov3"
sys.path.insert(0, str(ROOT))
from search import DinoEmbedder, load_config, load_excluded_trivial  # noqa: E402

EMB_DIM, PROJ_DIM, TEMPERATURE, SEED = 768, 512, 0.07, 42
EPOCHS, BATCH, REAL_PER_BATCH, LR, PATIENCE = 25, 256, 64, 1e-3, 5
K_VALUES = (1, 5, 10, 20)
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = load_config()

# ---------------- ortak veri: etiketler + donmuş ayrım
with open(ROOT / "data" / "eval" / "labels_clean.json", "r", encoding="utf-8-sig") as f:
    eslesme = json.load(f)["eslesme"]
with open(ROOT / "data" / "eval" / "projection_split.json", "r", encoding="utf-8-sig") as f:
    split = json.load(f)


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["embeddings"].astype(np.float32), d["filenames"].tolist()


def make_model():
    m = torch.nn.Sequential(
        torch.nn.Linear(EMB_DIM, EMB_DIM), torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(EMB_DIM, PROJ_DIM))
    return m.to(device)


def project_np(model, mat):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(mat), 4096):
            y = model(torch.from_numpy(mat[i:i + 4096]).to(device))
            outs.append(F.normalize(y, dim=1).cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def recall_at(qs, targets_list, cad_proj, cad_names):
    pos = {n: i for i, n in enumerate(cad_names)}
    hits = {k: 0 for k in K_VALUES}
    n = 0
    for q, targets in zip(qs, targets_list):
        tid = {pos[t] for t in targets if t in pos}
        if not tid:
            continue
        n += 1
        scores = cad_proj @ q
        top = np.argpartition(-scores, 20)[:20]
        top = top[np.argsort(-scores[top])]
        rank = next((r for r, i in enumerate(top, 1) if int(i) in tid), None)
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
    return {k: hits[k] / n for k in K_VALUES}, n


def train_and_test(tag, clean, hed, common, photo_embs):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cad_set = set(common)
    usable = {p: [Path(c).name for c in cads if Path(c).name in cad_set]
              for p, cads in eslesme.items()}
    usable = {p: c for p, c in usable.items() if c}
    name_pos = {n: i for i, n in enumerate(common)}
    photo_set = set(photo_embs)

    tr = [p for p in split["train"] if p in usable and p in photo_set]
    va = [p for p in split["val"] if p in usable and p in photo_set]
    te = [p for p in split["test"] if p in usable and p in photo_set]
    real_pairs = [(photo_embs[p], name_pos[c], c) for p in tr for c in usable[p]]
    print(f"\n[{tag}] eğitim {len(tr)} / val {len(va)} / test {len(te)} foto; "
          f"gerçek çift {len(real_pairs)}; sentetik {len(common)}")

    def subset(photos):
        return (np.stack([photo_embs[p] for p in photos]),
                [set(usable[p]) for p in photos])

    val_q, val_t = subset(va)
    test_q, test_t = subset(te)

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = max(1, len(common) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps)
    best = {"r10": -1, "epoch": -1, "state": None}
    patience = PATIENCE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(common))
        losses = []
        for s in range(steps):
            idx = order[s * BATCH:(s + 1) * BATCH]
            l_mat, r_mat = hed[idx], clean[idx]
            names = [common[i] for i in idx]
            picks = rng.integers(len(real_pairs), size=REAL_PER_BATCH)
            l_mat = np.concatenate([l_mat, np.stack([real_pairs[i][0] for i in picks])])
            r_mat = np.concatenate([r_mat, clean[[real_pairs[i][1] for i in picks]]])
            names += [real_pairs[i][2] for i in picks]
            zl = F.normalize(model(torch.from_numpy(l_mat).to(device)), dim=1)
            zr = F.normalize(model(torch.from_numpy(r_mat).to(device)), dim=1)
            logits = zl @ zr.T / TEMPERATURE
            arr = np.array(names)
            same = arr[:, None] == arr[None, :]
            np.fill_diagonal(same, False)
            logits = logits.masked_fill(torch.from_numpy(same).to(device), float("-inf"))
            labels = torch.arange(len(arr), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        cad_proj = project_np(model, clean)
        vr, _ = recall_at(project_np(model, val_q), val_t, cad_proj, common)
        print(f"[{tag}] epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
              f"val r@10 {vr[10]:.1%} r@20 {vr[20]:.1%}")
        if vr[10] > best["r10"]:
            best = {"r10": vr[10], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print(f"[{tag}] erken durdurma (en iyi epoch {best['epoch']})")
                break
    model.load_state_dict(best["state"])
    cad_proj = project_np(model, clean)
    tr_, n_te = recall_at(project_np(model, test_q), test_t, cad_proj, common)
    print(f"[{tag}] TEST ({n_te} foto): " +
          "  ".join(f"r@{k} {v:.1%}" for k, v in tr_.items()))
    return tr_


# ---------------- dinov3 tarafı
clean3, names3 = load_npz(HERE / "cad_clean_dinov3.npz")
hed3, names3b = load_npz(HERE / "cad_hed_dinov3.npz")
assert names3 == names3b
photos3, pnames3 = load_npz(HERE / "photos_dinov3.npz")
excl = load_excluded_trivial(cfg) if cfg.get("exclude_trivial") else set()
keep = [i for i, n in enumerate(names3) if n not in excl]
common3 = [names3[i] for i in keep]
r3 = train_and_test("dinov3", clean3[keep], hed3[keep], common3,
                    dict(zip(pnames3, photos3)))

# ---------------- dinov2 taban (aynı tarif; foto embed'leri cache'ten hesaplanır)
def load_side(variant_dir):
    emb = np.load(variant_dir / "cad_embeddings.npy").astype(np.float32)
    with open(variant_dir / "cad_filenames.json", "r", encoding="utf-8") as f:
        names = json.load(f)
    return emb, names

clean2, n2 = load_side(ROOT / "index")
hed2, n2b = load_side(ROOT / "index" / "variants" / "hedthick5_vitb14")
pos2 = {n: i for i, n in enumerate(n2)}
pos2b = {n: i for i, n in enumerate(n2b)}
common2 = [n for n in sorted(set(n2) & set(n2b)) if n not in excl]
clean2 = clean2[[pos2[n] for n in common2]]
hed2 = hed2[[pos2b[n] for n in common2]]

p2_path = HERE / "photos_dinov2.npz"
if p2_path.exists():
    photos2, pnames2 = load_npz(p2_path)
else:
    from PIL import Image
    embedder = DinoEmbedder(cfg, model_name="dinov2_vitb14")
    cache_dir = ROOT / "data" / "eval" / "cache"
    pnames2 = [p for p in sorted(eslesme) if (cache_dir / (p + ".lineart.png")).exists()]
    chunks = []
    for i in range(0, len(pnames2), 32):
        imgs = []
        for p in pnames2[i:i + 32]:
            with Image.open(cache_dir / (p + ".lineart.png")) as im:
                im.load()
                imgs.append(im.convert("RGB"))
        chunks.append(embedder.embed_batch(imgs))
    photos2 = np.concatenate(chunks, 0).astype(np.float32)
    np.savez(p2_path, embeddings=photos2, filenames=np.array(pnames2))
    del embedder
    torch.cuda.empty_cache()
r2 = train_and_test("dinov2", clean2, hed2, common2, dict(zip(pnames2, photos2)))

print("\n=== KIYAS (augmentsız, donmuş test) ===")
print("        " + "".join(f"  r@{k:<4}" for k in K_VALUES))
print("dinov2  " + "".join(f"  {100*r2[k]:5.1f}%" for k in K_VALUES))
print("dinov3  " + "".join(f"  {100*r3[k]:5.1f}%" for k in K_VALUES))
