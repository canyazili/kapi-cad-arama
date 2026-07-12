# -*- coding: utf-8 -*-
"""kapi_arama_app.py — Kapı fotoğrafından CAD çizimi arayan masaüstü uygulaması.

Fotoğraf seç -> pipeline (kapı-crop + metin silme + HED lineart) -> arama;
sonuçlar çizim görseli + dosya adı + benzerlik skoruyla ızgarada gösterilir.
Tek tık: dosya adını panoya kopyalar. Çift tık: görseli tam boyut açar.

PyInstaller paketi:
  pyinstaller kapi_arama_app.py --name KapiArama --noconsole ...
Exe, proje klasörü içinde durduğu sürece config/indeks/model dosyalarını
diskten bulur (search._find_root); indeks güncellenince exe'yi yeniden
derlemek GEREKMEZ.

Kendi kendini test: KapiArama.exe --selftest [foto_yolu]
  (arayüz açmadan tam pipeline'ı koşar, sonucu selftest_sonuc.txt'ye yazar)
"""
import os
import queue
import sys
import threading
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
THUMB = 190          # sonuç küçük resim kenarı (px)
COLS = 5             # ızgara sütun sayısı
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


class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("Kapı → CAD Arama")
        self.root.geometry("1280x820")
        self.engine = None
        self.jobs = queue.Queue()
        self._thumb_refs = []          # PhotoImage'lar GC'ye gitmesin

        bar = ttk.Frame(self.root, padding=8)
        bar.pack(fill="x")
        self.pick_btn = ttk.Button(bar, text="Fotoğraf Seç…", command=self.pick_photo)
        self.pick_btn.pack(side="left")
        ttk.Label(bar, text="  Sonuç sayısı:").pack(side="left")
        self.k_var = tk.IntVar(value=20)
        ttk.Spinbox(bar, from_=5, to=50, increment=5, width=4,
                    textvariable=self.k_var).pack(side="left")
        self.group_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Benzer çizimleri grupla",
                        variable=self.group_var).pack(side="left", padx=(12, 0))
        self.status = tk.StringVar(value="Hazır. Bir kapı fotoğrafı seçin.")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=16)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)

        # sol: sorgu önizlemesi
        left = ttk.Frame(body, padding=8, width=240)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        ttk.Label(left, text="Fotoğraf").pack()
        self.photo_lbl = ttk.Label(left)
        self.photo_lbl.pack(pady=(0, 8))
        ttk.Label(left, text="İşlenmiş (lineart)").pack()
        self.lineart_lbl = ttk.Label(left)
        self.lineart_lbl.pack()

        # sağ: kaydırılabilir sonuç ızgarası
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, highlightthickness=0)
        vsb = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(-e.delta // 120, "units"))

        self.root.after(100, self._poll_jobs)

    # ------------------------------------------------------------------ akış
    def pick_photo(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Kapı fotoğrafı seçin",
            filetypes=[("Görseller", "*.png *.jpg *.jpeg *.webp"), ("Tümü", "*.*")])
        if not path:
            return
        self.pick_btn.state(["disabled"])
        self.show_query_photo(Path(path))
        self.status.set("Model yükleniyor…" if self.engine is None else "Aranıyor…")
        threading.Thread(target=self._search_worker,
                         args=(Path(path), int(self.k_var.get()),
                               bool(self.group_var.get())), daemon=True).start()

    def _search_worker(self, photo: Path, k: int, group: bool):
        try:
            if self.engine is None:
                self.engine = search.get_engine()
            lineart, _, _ = self.engine.prepare_query(photo)
            if group:
                raw = self.engine.search_prepared(lineart, k=min(k * 3, DUP_FETCH_CAP))
                results = group_results(raw, k)
            else:
                results = [(n, s, []) for n, s in self.engine.search_prepared(lineart, k=k)]
            self.jobs.put(("done", lineart, results))
        except Exception:
            self.jobs.put(("error", traceback.format_exc(), None))

    def _poll_jobs(self):
        try:
            while True:
                kind, a, b = self.jobs.get_nowait()
                if kind == "done":
                    self.show_lineart(a)
                    self.show_results(b)
                    n_var = sum(len(v) for _, _, v in b)
                    ek = f" (+{n_var} benzer varyant)" if n_var else ""
                    self.status.set(f"{len(b)} sonuç{ek}. Tık: adı kopyala, çift tık: aç.")
                else:
                    self.status.set("HATA — ayrıntı konsol/logda.")
                    print(a)
                self.pick_btn.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_jobs)

    # ------------------------------------------------------------------ görünüm
    def _photoimage(self, pil_img, side):
        from PIL import ImageTk
        img = pil_img.copy()
        img.thumbnail((side, side))
        ph = ImageTk.PhotoImage(img)
        self._thumb_refs.append(ph)
        return ph

    def show_query_photo(self, path: Path):
        from PIL import Image
        with Image.open(path) as im:
            im.load()
            ph = self._photoimage(im.convert("RGB"), 220)
        self.photo_lbl.configure(image=ph)
        self.lineart_lbl.configure(image="")

    def show_lineart(self, lineart):
        ph = self._photoimage(lineart, 220)
        self.lineart_lbl.configure(image=ph)

    def show_results(self, results):
        from PIL import Image
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._thumb_refs = self._thumb_refs[-2:]   # sorgu önizlemeleri kalsın
        for i, (name, score, variants) in enumerate(results):
            cell = self.ttk.Frame(self.grid_frame, padding=6)
            cell.grid(row=i // COLS, column=i % COLS, sticky="n")
            p = cad_image_path(name)
            try:
                with Image.open(p) as im:
                    im.load()
                    ph = self._photoimage(im.convert("RGB"), THUMB)
                img_lbl = self.tk.Label(cell, image=ph, cursor="hand2",
                                        relief="solid", borderwidth=1)
            except Exception:
                img_lbl = self.tk.Label(cell, text="(görsel açılamadı)",
                                        width=24, height=10)
            img_lbl.pack()
            self.tk.Label(cell, text=f"{i + 1}. {score:.3f}").pack()
            name_lbl = self.tk.Label(cell, text=name, wraplength=THUMB,
                                     fg="#0a58ca", cursor="hand2")
            name_lbl.pack()
            for w in (img_lbl, name_lbl):
                w.bind("<Button-1>", lambda e, n=name: self.copy_name(n))
                w.bind("<Double-Button-1>", lambda e, pp=p: self.open_image(pp))
            if variants:
                var_lbl = self.tk.Label(cell, text=f"+{len(variants)} benzer varyant",
                                        fg="#6c757d", cursor="hand2",
                                        font=("TkDefaultFont", 9, "underline"))
                var_lbl.pack()
                var_lbl.bind("<Button-1>",
                             lambda e, n=name, vs=variants: self.show_variants(n, vs))

    def show_variants(self, rep_name, variants):
        """Bir temsilcinin ikiz varyantlarını ayrı pencerede listeler."""
        from PIL import Image
        win = self.tk.Toplevel(self.root)
        win.title(f"{rep_name} — {len(variants)} benzer varyant")
        win.geometry("1020x640")
        win._thumb_refs = []           # pencereye özel referanslar (GC koruması)

        canvas = self.tk.Canvas(win, highlightthickness=0)
        vsb = self.ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = self.ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        for i, (name, score) in enumerate(variants):
            cell = self.ttk.Frame(frame, padding=6)
            cell.grid(row=i // COLS, column=i % COLS, sticky="n")
            p = cad_image_path(name)
            try:
                with Image.open(p) as im:
                    im.load()
                    img = im.convert("RGB")
                img.thumbnail((THUMB, THUMB))
                from PIL import ImageTk
                ph = ImageTk.PhotoImage(img)
                win._thumb_refs.append(ph)
                img_lbl = self.tk.Label(cell, image=ph, cursor="hand2",
                                        relief="solid", borderwidth=1)
            except Exception:
                img_lbl = self.tk.Label(cell, text="(görsel açılamadı)",
                                        width=24, height=10)
            img_lbl.pack()
            self.tk.Label(cell, text=f"{score:.3f}").pack()
            name_lbl = self.tk.Label(cell, text=name, wraplength=THUMB,
                                     fg="#0a58ca", cursor="hand2")
            name_lbl.pack()
            for w in (img_lbl, name_lbl):
                w.bind("<Button-1>", lambda e, n=name: self.copy_name(n))
                w.bind("<Double-Button-1>", lambda e, pp=p: self.open_image(pp))

    def copy_name(self, name):
        self.root.clipboard_clear()
        self.root.clipboard_append(name)
        self.status.set(f"Panoya kopyalandı: {name}")

    def open_image(self, path: Path):
        import os
        if path.exists():
            os.startfile(path)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        arg = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
        sys.exit(selftest(arg))
    App().run()
