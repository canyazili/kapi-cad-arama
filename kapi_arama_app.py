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

# ------------------------------------------------------------------ tema renkleri
APP_BG = "#eef2f7"        # genel zemin (açık gri-mavi)
CARD_BG = "#ffffff"       # kart zemini
CARD_BORDER = "#d8dfea"   # kart kenarlığı
HEADER_BG = "#1f3a5f"     # üst şerit (koyu lacivert)
HEADER_FG = "#ffffff"
HEADER_SUB = "#a8c0dc"
ACCENT = "#2b6cb0"        # ana vurgu (mavi)
ACCENT_ACTIVE = "#1f5591"
TEXT = "#1f2933"
MUTED = "#68778c"
LINK = "#2b6cb0"
DANGER = "#c0392b"
OK_GREEN = "#1e7e34"
FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI Semibold", 10)
FONT_TITLE = ("Segoe UI Semibold", 15)


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
        self.root.geometry("1280x860")
        self.root.configure(bg=APP_BG)
        self.engine = None
        self.jobs = queue.Queue()
        self._thumb_refs = []          # PhotoImage'lar GC'ye gitmesin
        self._setup_style()

        # Üst şerit: uygulama adı
        header = tk.Frame(self.root, bg=HEADER_BG)
        header.pack(fill="x")
        tk.Frame(header, bg=ACCENT, height=3).pack(fill="x", side="bottom")
        tk.Label(header, text="Kapı → CAD Arama", bg=HEADER_BG, fg=HEADER_FG,
                 font=FONT_TITLE).pack(side="left", padx=16, pady=(10, 10))
        tk.Label(header, text="fotoğraftan en benzer AutoCAD çizimini bulur",
                 bg=HEADER_BG, fg=HEADER_SUB, font=FONT).pack(
                     side="left", padx=(0, 16), pady=(14, 10))

        # İki sekme: Arama (mevcut akış) + Kapı Ekle (katalog büyütme)
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(8, 0))
        self._nb = nb
        search_tab = ttk.Frame(nb)
        add_tab = ttk.Frame(nb)
        recent_tab = ttk.Frame(nb)
        nb.add(search_tab, text="  🔍  Arama  ")
        nb.add(add_tab, text="  ➕  Kapı Ekle  ")
        nb.add(recent_tab, text="  🕒  Son Eklenenler  ")
        nb.bind("<<NotebookTabChanged>>",
                lambda e: self._recent_refresh() if nb.index("current") == 2 else None)

        bar = ttk.Frame(search_tab, padding=10)
        bar.pack(fill="x")
        self.pick_btn = ttk.Button(bar, text="📷  Fotoğraf Seç…",
                                   style="Accent.TButton", command=self.pick_photo)
        self.pick_btn.pack(side="left")
        ttk.Label(bar, text="  Sonuç sayısı:").pack(side="left", padx=(14, 4))
        self.k_var = tk.IntVar(value=20)
        ttk.Spinbox(bar, from_=5, to=50, increment=5, width=4,
                    textvariable=self.k_var).pack(side="left")
        self.group_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Benzer çizimleri grupla",
                        variable=self.group_var).pack(side="left", padx=(14, 0))
        self.status = tk.StringVar(value="Hazır. Bir kapı fotoğrafı seçin.")
        ttk.Label(bar, textvariable=self.status, foreground=MUTED).pack(
            side="left", padx=16)

        body = ttk.Frame(search_tab)
        body.pack(fill="both", expand=True)

        # sol: sorgu önizlemesi (kart görünümü)
        left = ttk.Frame(body, padding=(10, 4, 6, 10), width=260)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        for baslik, attr in (("Fotoğraf", "photo_lbl"),
                             ("İşlenmiş (çizgi hali)", "lineart_lbl")):
            kart = tk.Frame(left, bg=CARD_BG, highlightbackground=CARD_BORDER,
                            highlightthickness=1)
            kart.pack(fill="x", pady=(0, 10))
            tk.Label(kart, text=baslik, bg=CARD_BG, fg=MUTED,
                     font=FONT_BOLD).pack(anchor="w", padx=8, pady=(6, 0))
            lbl = tk.Label(kart, bg=CARD_BG)
            lbl.pack(padx=8, pady=8)
            setattr(self, attr, lbl)
            if attr == "photo_lbl":     # seçilen fotoğrafın dosya adı
                self.photo_name_lbl = tk.Label(kart, bg=CARD_BG, fg=ACCENT,
                                               font=FONT_SMALL, wraplength=230)
                self.photo_name_lbl.pack(padx=8, pady=(0, 8))

        # sağ: kaydırılabilir sonuç ızgarası
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, highlightthickness=0, bg=APP_BG)
        vsb = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # Tekerlek, imlecin üstünde durduğu kaydırılabilir alanı kaydırsın
        self._wheel_canvases = [self.canvas]
        self.root.bind_all("<MouseWheel>", self._on_wheel)

        self._build_add_tab(add_tab)
        self._build_recent_tab(recent_tab)
        self.root.after(100, self._poll_jobs)

    def _setup_style(self):
        """ttk temasını modern açık renk paletiyle kurar."""
        s = self.ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=APP_BG, foreground=TEXT, font=FONT)
        s.configure("TFrame", background=APP_BG)
        s.configure("TLabel", background=APP_BG, foreground=TEXT)
        s.configure("TNotebook", background=APP_BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(18, 8), font=FONT_BOLD,
                    background="#dbe3ee", foreground=MUTED)
        s.map("TNotebook.Tab",
              background=[("selected", CARD_BG)],
              foreground=[("selected", ACCENT)])
        s.configure("TButton", padding=(12, 6), background=CARD_BG,
                    foreground=TEXT, bordercolor=CARD_BORDER, relief="flat")
        s.map("TButton", background=[("active", "#e7edf5")])
        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    bordercolor=ACCENT, focuscolor=ACCENT)
        s.map("Accent.TButton",
              background=[("disabled", "#9db8d2"), ("active", ACCENT_ACTIVE)],
              foreground=[("disabled", "#f0f0f0")])
        s.configure("TCheckbutton", background=APP_BG, foreground=TEXT)
        s.map("TCheckbutton", background=[("active", APP_BG)])
        s.configure("TSpinbox", fieldbackground=CARD_BG, background=CARD_BG,
                    bordercolor=CARD_BORDER, arrowcolor=TEXT)
        s.configure("Vertical.TScrollbar", background="#c3cedd",
                    troughcolor=APP_BG, bordercolor=APP_BG, arrowcolor=MUTED)
        s.configure("Col.TLabelframe", background=CARD_BG,
                    bordercolor=CARD_BORDER, relief="solid")
        s.configure("Col.TLabelframe.Label", background=APP_BG,
                    foreground=ACCENT, font=FONT_BOLD)

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

    def _on_wheel(self, event):
        """Tekerlek: imlecin üstünde olduğu kaydırılabilir alanı bul ve kaydır."""
        w = self.root.winfo_containing(event.x_root, event.y_root)
        while w is not None:
            if w in self._wheel_canvases:
                w.yview_scroll(-event.delta // 120, "units")
                return
            w = getattr(w, "master", None)
        if self._wheel_canvases:
            self._wheel_canvases[0].yview_scroll(-event.delta // 120, "units")

    # ------------------------------------------------------------- Kapı Ekle sekmesi
    def _build_add_tab(self, tab):
        """Solda kapı fotoğrafları, sağda CAD çizimleri; ikisi de zorunlu."""
        tk, ttk = self.tk, self.ttk
        self.add_photos = []     # seçilen fotoğraf yolları
        self.add_drawings = []   # seçilen çizim yolları
        self._add_thumb_refs = []

        info = ttk.Label(tab, padding=(12, 10), foreground=MUTED, wraplength=1200,
                         text="Yeni kapıyı kataloğa ekle: solda kapının FOTOĞRAFLARINI, "
                              "sağda ÇİZİMLERİNİ seç. Çizim olarak doğrudan AutoCAD "
                              "dosyası (DWG/DXF) verebilirsin — PNG'ye otomatik çevrilir; "
                              "PNG/JPG export da olur. Fotoğraf ve çizim İKİSİ DE "
                              "zorunludur, kapı katalogda çizim dosyasının adıyla görünür. "
                              "Tek seferde TEK kapı eklenir (aynı kapının birden çok "
                              "fotoğrafı ve çizim varyantı olabilir).")
        info.pack(fill="x")

        cols = ttk.Frame(tab)
        cols.pack(fill="both", expand=True)
        self._add_panels = {}
        for key, title, btn_text in (
                ("foto", "📷 Kapı Fotoğrafları", "Fotoğraf Ekle…"),
                ("cizim", "📐 CAD Çizimleri", "Çizim Ekle…")):
            col = ttk.LabelFrame(cols, text=f" {title} ", padding=8,
                                 style="Col.TLabelframe")
            col.pack(side="left", fill="both", expand=True, padx=8, pady=4)
            ttk.Button(col, text=btn_text, style="Accent.TButton",
                       command=lambda k=key: self._add_pick_files(k)).pack(anchor="w")
            canvas = tk.Canvas(col, highlightthickness=0, bg=CARD_BG)
            vsb = ttk.Scrollbar(col, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True, pady=(8, 0))
            frame = tk.Frame(canvas, bg=CARD_BG)
            canvas.create_window((0, 0), window=frame, anchor="nw")
            frame.bind("<Configure>", lambda e, c=canvas:
                       c.configure(scrollregion=c.bbox("all")))
            self._wheel_canvases.append(canvas)
            self._add_panels[key] = frame

        alt = ttk.Frame(tab, padding=10)
        alt.pack(fill="x")
        self.add_status = tk.StringVar(value="Hazır.")
        self.add_go_btn = ttk.Button(alt, text="✔  Kataloğa Ekle",
                                     style="Accent.TButton", command=self._start_add)
        self.add_go_btn.pack(side="left")
        ttk.Button(alt, text="Listeyi Temizle",
                   command=self._add_clear).pack(side="left", padx=8)
        ttk.Label(alt, textvariable=self.add_status,
                  foreground=ACCENT).pack(side="left", padx=12)

    def _add_pick_files(self, key):
        from tkinter import filedialog
        if key == "foto":
            title = "Kapı fotoğraflarını seçin"
            types = [("Görseller", "*.png *.jpg *.jpeg *.webp"), ("Tümü", "*.*")]
        else:
            title = "CAD çizimlerini seçin (DWG/DXF ya da PNG/JPG export)"
            types = [("CAD / Görsel", "*.dwg *.dxf *.png *.jpg *.jpeg *.webp"),
                     ("AutoCAD", "*.dwg *.dxf"), ("Tümü", "*.*")]
        paths = filedialog.askopenfilenames(title=title, filetypes=types)
        target = self.add_photos if key == "foto" else self.add_drawings
        for p in paths:
            if p not in target:
                target.append(p)
        self._add_refresh(key)

    def _add_refresh(self, key):
        """Seçilen dosyaların küçük resimlerini yeniden çizer (tık: listeden çıkar)."""
        from PIL import Image
        frame = self._add_panels[key]
        target = self.add_photos if key == "foto" else self.add_drawings
        for w in frame.winfo_children():
            w.destroy()
        per_row = 3
        for i, p in enumerate(list(target)):
            cell = self.tk.Frame(frame, bg=CARD_BG, highlightbackground=CARD_BORDER,
                                 highlightthickness=1)
            cell.grid(row=i // per_row, column=i % per_row, sticky="n",
                      padx=4, pady=4)
            if Path(p).suffix.lower() in (".dwg", ".dxf"):
                lbl = self.tk.Label(cell, text="📐\nAutoCAD çizimi\n(eklenirken PNG'ye\nçevrilir)",
                                    bg=CARD_BG, fg=MUTED, width=18, height=8,
                                    font=FONT_SMALL, cursor="hand2")
            else:
                try:
                    with Image.open(p) as im:
                        im.load()
                        img = im.convert("RGB")
                    img.thumbnail((150, 150))
                    from PIL import ImageTk
                    ph = ImageTk.PhotoImage(img)
                    self._add_thumb_refs.append(ph)
                    lbl = self.tk.Label(cell, image=ph, bg=CARD_BG, cursor="hand2")
                except Exception:
                    lbl = self.tk.Label(cell, text="(açılamadı)", bg=CARD_BG,
                                        fg=MUTED, width=18, height=8)
            lbl.pack(padx=4, pady=(4, 2))
            self.tk.Label(cell, text=Path(p).name, wraplength=150,
                          bg=CARD_BG, fg=TEXT, font=FONT_SMALL).pack()
            ck = self.tk.Label(cell, text="✕ çıkar", bg=CARD_BG, fg=DANGER,
                               cursor="hand2", font=FONT_SMALL)
            ck.pack(pady=(0, 4))
            for w in (lbl, ck):
                w.bind("<Button-1>", lambda e, pp=p, k=key: self._add_remove(pp, k))
        say = f"{len(self.add_photos)} fotoğraf, {len(self.add_drawings)} çizim seçili."
        self.add_status.set(say)

    def _add_remove(self, path, key):
        target = self.add_photos if key == "foto" else self.add_drawings
        if path in target:
            target.remove(path)
        self._add_refresh(key)

    def _add_clear(self):
        self.add_photos.clear()
        self.add_drawings.clear()
        self._add_thumb_refs.clear()
        self._add_refresh("foto")
        self._add_refresh("cizim")
        self.add_status.set("Liste temizlendi.")

    def _start_add(self):
        if not self.add_photos or not self.add_drawings:
            self.add_status.set("Eksik: en az bir FOTOĞRAF ve bir ÇİZİM seçmelisin "
                                f"(şu an {len(self.add_photos)} foto, "
                                f"{len(self.add_drawings)} çizim).")
            return
        self.add_go_btn.state(["disabled"])
        self.add_status.set("Ekleniyor… (ilk eklemede model yüklemesi sürebilir)")
        threading.Thread(target=self._add_worker,
                         args=(list(self.add_drawings), list(self.add_photos)),
                         daemon=True).start()

    def _add_worker(self, drawings, photos):
        import katalog
        try:
            if self.engine is None:
                self.jobs.put(("add_status", "Model yükleniyor…", None))
                self.engine = search.get_engine()
            msg = katalog.add_entry(
                drawings, photos, engine=self.engine,
                log=lambda m: self.jobs.put(("add_status", m, None)))
            global _base_emb
            _base_emb = None            # base embedding önbelleği yenilensin (gruplama)
            self.jobs.put(("add_done", "TAMAM: " + msg, True))
        except Exception as e:
            traceback.print_exc()
            self.jobs.put(("add_done", f"HATA: {e}", False))

    # -------------------------------------------------------- Son Eklenenler sekmesi
    def _build_recent_tab(self, tab):
        tk, ttk = self.tk, self.ttk
        ust = ttk.Frame(tab, padding=(12, 10))
        ust.pack(fill="x")
        ttk.Label(ust, foreground=MUTED, wraplength=1200,
                  text="Uygulamadan eklenen kapılar (yeniden eskiye). Yanlış "
                       "eklediysen 'Sil' ile hem çizimleri hem fotoğrafları hem de "
                       "eşleşme kayıtlarını tamamen geri alırsın. Orijinal katalog "
                       "buradan silinemez.").pack(side="left")
        self.recent_status = tk.StringVar(value="")
        ttk.Label(ust, textvariable=self.recent_status,
                  foreground=ACCENT).pack(side="right")

        canvas = tk.Canvas(tab, highlightthickness=0, bg=APP_BG)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self._recent_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self._recent_frame, anchor="nw")
        self._recent_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._wheel_canvases.append(canvas)
        self._recent_thumb_refs = []

    def _recent_refresh(self):
        from PIL import Image, ImageTk
        import katalog
        for w in self._recent_frame.winfo_children():
            w.destroy()
        self._recent_thumb_refs.clear()
        entries = katalog.list_entries()
        if not entries:
            self.tk.Label(self._recent_frame, text="Henüz uygulamadan kapı eklenmedi.",
                          bg=APP_BG, fg=MUTED, font=FONT).pack(padx=20, pady=20)
            self.recent_status.set("")
            return
        self.recent_status.set(f"{len(entries)} ekleme")
        for entry in entries:
            kart = self.tk.Frame(self._recent_frame, bg=CARD_BG,
                                 highlightbackground=CARD_BORDER, highlightthickness=1)
            kart.pack(fill="x", padx=6, pady=5)
            solda = self.tk.Frame(kart, bg=CARD_BG)
            solda.pack(side="left", padx=8, pady=8)
            for ph in entry.get("fotolar", [])[:4]:
                p = katalog.ADDED_PHOTOS_DIR / ph
                try:
                    with Image.open(p) as im:
                        im.load()
                        img = im.convert("RGB")
                    img.thumbnail((110, 110))
                    tkimg = ImageTk.PhotoImage(img)
                    self._recent_thumb_refs.append(tkimg)
                    self.tk.Label(solda, image=tkimg, bg=CARD_BG).pack(
                        side="left", padx=3)
                except Exception:
                    self.tk.Label(solda, text="(foto yok)", bg=CARD_BG,
                                  fg=MUTED, width=12, height=6).pack(side="left")
            cizimler = entry.get("cizimler", [])
            fotolar = entry.get("fotolar", [])
            orta = self.tk.Frame(kart, bg=CARD_BG)
            orta.pack(side="left", padx=10, pady=8, fill="x", expand=True)
            self.tk.Label(orta, bg=CARD_BG, fg=TEXT, font=FONT_BOLD, justify="left",
                          text=(f"{entry.get('tarih', '?')}  —  {len(fotolar)} "
                                f"fotoğraf, {len(cizimler)} çizim eklendi")).pack(
                                    anchor="w")
            for cz in cizimler:
                satir = self.tk.Frame(orta, bg=CARD_BG)
                satir.pack(anchor="w")
                self.tk.Label(satir, text="• " + cz, bg=CARD_BG, fg=TEXT,
                              font=FONT_SMALL).pack(side="left")
                x = self.tk.Label(satir, text=" ✕", bg=CARD_BG, fg=DANGER,
                                  cursor="hand2", font=FONT_SMALL)
                x.pack(side="left")
                x.bind("<Button-1>", lambda e, en=entry, c=cz:
                       self._recent_delete_drawing(en, c))
            btn = self.ttk.Button(kart, text="Tümünü Sil",
                                  command=lambda en=entry: self._recent_delete(en))
            btn.pack(side="right", padx=10)

    def _recent_delete(self, entry):
        from tkinter import messagebox
        cizimler = ", ".join(entry.get("cizimler", []))
        if not messagebox.askyesno(
                "Silme onayı",
                f"Bu ekleme tamamen geri alınacak:\n\n{cizimler}\n\n"
                "Çizimler arama kataloğundan, fotoğraflar ve eşleşme kayıtları "
                "diskten silinecek. Emin misin?"):
            return
        self.recent_status.set("Siliniyor…")
        threading.Thread(target=self._recent_delete_worker,
                         args=(entry,), daemon=True).start()

    def _recent_delete_worker(self, entry):
        import katalog
        try:
            msg = katalog.remove_entry(entry, engine=self.engine, log=print)
            global _base_emb
            _base_emb = None
            self.jobs.put(("recent_done", msg, None))
        except Exception as e:
            traceback.print_exc()
            self.jobs.put(("recent_done", f"HATA: {e}", None))

    def _recent_delete_drawing(self, entry, drawing_name):
        from tkinter import messagebox
        kalan = len(entry.get("cizimler", [])) - 1
        ek = ("\n\nBu kayıttaki SON çizim: fotoğraflarıyla birlikte ekleme "
              "komple geri alınacak." if kalan == 0 else
              f"\n\nKayıttaki diğer {kalan} çizim ve fotoğraflar kalacak.")
        if not messagebox.askyesno(
                "Çizim silme onayı",
                f"'{drawing_name}' katalogdan silinecek.{ek}\nEmin misin?"):
            return
        self.recent_status.set("Siliniyor…")
        threading.Thread(target=self._recent_delete_drawing_worker,
                         args=(entry, drawing_name), daemon=True).start()

    def _recent_delete_drawing_worker(self, entry, drawing_name):
        import katalog
        try:
            msg = katalog.remove_drawing(entry, drawing_name,
                                         engine=self.engine, log=print)
            global _base_emb
            _base_emb = None
            self.jobs.put(("recent_done", msg, None))
        except Exception as e:
            traceback.print_exc()
            self.jobs.put(("recent_done", f"HATA: {e}", None))

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
                if kind == "recent_done":
                    self.recent_status.set(a)
                    self._recent_refresh()
                elif kind == "add_status":
                    self.add_status.set(a)
                elif kind == "add_done":
                    self.add_status.set(a)
                    self.add_go_btn.state(["!disabled"])
                    if b:                  # başarılıysa seçim listelerini boşalt
                        self.add_photos.clear()
                        self.add_drawings.clear()
                        self._add_refresh("foto")
                        self._add_refresh("cizim")
                        self.add_status.set(a)
                    self.status.set(a)
                elif kind == "done":
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
        self.photo_name_lbl.configure(text=path.name)
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
            cell = self.tk.Frame(self.grid_frame, bg=CARD_BG,
                                 highlightbackground=CARD_BORDER,
                                 highlightthickness=1)
            cell.grid(row=i // COLS, column=i % COLS, sticky="n", padx=5, pady=5)
            p = cad_image_path(name)
            try:
                with Image.open(p) as im:
                    im.load()
                    ph = self._photoimage(im.convert("RGB"), THUMB)
                img_lbl = self.tk.Label(cell, image=ph, cursor="hand2", bg=CARD_BG)
            except Exception:
                img_lbl = self.tk.Label(cell, text="(görsel açılamadı)", bg=CARD_BG,
                                        fg=MUTED, width=24, height=10)
            img_lbl.pack(padx=6, pady=(6, 2))
            renk = OK_GREEN if score >= 0.75 else (ACCENT if score >= 0.55 else MUTED)
            self.tk.Label(cell, text=f"{i + 1}.  benzerlik {score:.3f}",
                          bg=CARD_BG, fg=renk, font=FONT_SMALL).pack()
            name_lbl = self.tk.Label(cell, text=name, wraplength=THUMB, bg=CARD_BG,
                                     fg=LINK, cursor="hand2", font=FONT_BOLD)
            name_lbl.pack(padx=6)
            for w in (img_lbl, name_lbl):
                w.bind("<Button-1>", lambda e, n=name: self.copy_name(n))
                w.bind("<Double-Button-1>", lambda e, pp=p: self.open_image(pp))
            if variants:
                var_lbl = self.tk.Label(cell, text=f"+{len(variants)} benzer varyant",
                                        bg=CARD_BG, fg=MUTED, cursor="hand2",
                                        font=("Segoe UI", 9, "underline"))
                var_lbl.bind("<Button-1>",
                             lambda e, n=name, vs=variants: self.show_variants(n, vs))
                var_lbl.pack(pady=(0, 6))
            else:
                self.tk.Frame(cell, bg=CARD_BG, height=6).pack()

    def show_variants(self, rep_name, variants):
        """Bir temsilcinin ikiz varyantlarını ayrı pencerede listeler."""
        from PIL import Image
        win = self.tk.Toplevel(self.root)
        win.title(f"{rep_name} — {len(variants)} benzer varyant")
        win.geometry("1020x640")
        win.configure(bg=APP_BG)
        win._thumb_refs = []           # pencereye özel referanslar (GC koruması)

        canvas = self.tk.Canvas(win, highlightthickness=0, bg=APP_BG)
        vsb = self.ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = self.ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        for i, (name, score) in enumerate(variants):
            cell = self.tk.Frame(frame, bg=CARD_BG,
                                 highlightbackground=CARD_BORDER,
                                 highlightthickness=1)
            cell.grid(row=i // COLS, column=i % COLS, sticky="n", padx=5, pady=5)
            p = cad_image_path(name)
            try:
                with Image.open(p) as im:
                    im.load()
                    img = im.convert("RGB")
                img.thumbnail((THUMB, THUMB))
                from PIL import ImageTk
                ph = ImageTk.PhotoImage(img)
                win._thumb_refs.append(ph)
                img_lbl = self.tk.Label(cell, image=ph, cursor="hand2", bg=CARD_BG)
            except Exception:
                img_lbl = self.tk.Label(cell, text="(görsel açılamadı)", bg=CARD_BG,
                                        fg=MUTED, width=24, height=10)
            img_lbl.pack(padx=6, pady=(6, 2))
            self.tk.Label(cell, text=f"benzerlik {score:.3f}", bg=CARD_BG,
                          fg=MUTED, font=FONT_SMALL).pack()
            name_lbl = self.tk.Label(cell, text=name, wraplength=THUMB, bg=CARD_BG,
                                     fg=LINK, cursor="hand2", font=FONT_BOLD)
            name_lbl.pack(padx=6, pady=(0, 6))
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
