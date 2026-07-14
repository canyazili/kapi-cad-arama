# -*- coding: utf-8 -*-
"""kapi_arama_flet.py — Kapı → CAD Arama, Flet arayüzü (Tkinter'ın yerine).

Üç sekme: Arama / Kapı Ekle / Son Eklenenler — işlevler kapi_arama_app.py
(Tkinter) ile birebir; arama çekirdeği search.py, katalog işleri katalog.py.

Çalıştırma:  python kapi_arama_flet.py [--demo]
  --demo: açılışta örnek foto yükler ve gerçek aramayı kendiliğinden koşar.
Kendi kendini test:  python kapi_arama_flet.py --selftest [foto]
  (arayüz açmadan tam pipeline; sonuç selftest_sonuc.txt — eski exe ile aynı)
"""
import base64
import io
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import flet as ft

import kapi_arama_app as core   # ROOT çözümü, group_results, portable cache kurulumu
import search

ROOT = search.ROOT
DEMO = "--demo" in sys.argv
PHOTO_EXTS = ["png", "jpg", "jpeg", "webp"]
DRAWING_EXTS = ["dwg", "dxf"] + PHOTO_EXTS

# ------------------------------------------------------------------ palet
BG = "#F7F8FA"          # genel zemin
CARD = "#FFFFFF"        # kart yüzeyi
NAVY = "#1E3A5F"        # ana renk (üst bar, birincil buton, aktif sekme)
AMBER = "#E8A33D"       # vurgu (yüksek skor rozeti, yıldız) — az kullan
NAVY_SOFT = "#E9F0F7"   # hover / seçim zemini
TEXT = "#1A1A1A"
MUTED = "#6B7280"
BORDER = "#E3E7EE"
TAB_INACTIVE = "#A8C0DC"
DANGER = "#C0392B"
OK_GREEN = "#1E7E34"

SHADOW = ft.BoxShadow(blur_radius=6, spread_radius=0,
                      color=ft.Colors.with_opacity(0.07, "black"),
                      offset=ft.Offset(0, 2))
SHADOW_HOVER = ft.BoxShadow(blur_radius=14, spread_radius=1,
                            color=ft.Colors.with_opacity(0.14, "black"),
                            offset=ft.Offset(0, 4))
W600 = ft.FontWeight.W_600


# ------------------------------------------------------- görsel yardımcıları
def pil_to_b64(im) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def b64_image(path: Path, max_side=560) -> str | None:
    """Görseli beyaz zemine bindirip küçültür, base64 döner."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, "white")
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            return pil_to_b64(im)
    except Exception:
        return None


def dashed_zone_b64(w=268, h=190, r=14) -> str:
    """Kesikli lacivert çerçeveli boş alan görseli (PIL ile, 2x keskinlik)."""
    from PIL import Image, ImageDraw
    s = 2
    W, H, R = w * s, h * s, r * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = (30, 58, 95, 255)
    lw, dash, gap = 2 * s, 10 * s, 7 * s
    for box, a0, a1 in (((0, 0, 2 * R, 2 * R), 180, 270),
                        ((W - 2 * R - lw, 0, W - lw, 2 * R), 270, 360),
                        ((W - 2 * R - lw, H - 2 * R - lw, W - lw, H - lw), 0, 90),
                        ((0, H - 2 * R - lw, 2 * R, H - lw), 90, 180)):
        d.arc(box, a0, a1, fill=col, width=lw)
    def dline(x0, y0, x1, y1):
        L = max(abs(x1 - x0), abs(y1 - y0))
        pos = 0
        while pos < L:
            seg = min(dash, L - pos)
            t0, t1 = pos / L, (pos + seg) / L
            d.line((x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                    x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1), fill=col, width=lw)
            pos += dash + gap
    dline(R, lw // 2, W - R, lw // 2)
    dline(R, H - lw // 2 - 1, W - R, H - lw // 2 - 1)
    dline(lw // 2, R, lw // 2, H - R)
    dline(W - lw // 2 - 1, R, W - lw // 2 - 1, H - R)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ================================================================== uygulama
class FletApp:
    def __init__(self, page: ft.Page):
        self.page = page
        page.title = "Kapı → CAD Arama"
        page.padding = 0
        page.bgcolor = BG
        page.window.min_width = 1280
        page.window.min_height = 800
        page.window.maximized = True

        self.engine = None
        self.engine_lock = threading.Lock()
        self.photo_path: Path | None = None
        self.selected_card = None
        self.k = 20
        self.group = True
        self.add_photos: list[str] = []
        self.add_drawings: list[str] = []

        self.photo_picker = ft.FilePicker(on_result=self._photo_picked)
        self.add_foto_picker = ft.FilePicker(
            on_result=lambda e: self._add_picked(e, "foto"))
        self.add_cizim_picker = ft.FilePicker(
            on_result=lambda e: self._add_picked(e, "cizim"))
        page.overlay.extend([self.photo_picker, self.add_foto_picker,
                             self.add_cizim_picker])

        self.tab_defs = ["Arama", "Kapı Ekle", "Son Eklenenler"]
        self.tab_bars = []
        self.content = ft.Container(expand=True)
        page.add(ft.Column([self._topbar(), self.content], spacing=0, expand=True))

        self.search_view = self._build_search_view()
        self.add_view = self._build_add_view()
        self.recent_view = self._build_recent_view()
        start_tab = 0
        if "--tab" in sys.argv:
            start_tab = int(sys.argv[sys.argv.index("--tab") + 1])
        self._show_tab(start_tab)
        if DEMO:
            threading.Thread(target=self._demo_flow, daemon=True).start()

    def _get_engine(self):
        with self.engine_lock:
            if self.engine is None:
                self.engine = search.get_engine()
            return self.engine

    # ------------------------------------------------------------ üst bar
    def _topbar(self):
        tabs = []
        for i, name in enumerate(self.tab_defs):
            t = ft.Container(
                height=56, alignment=ft.alignment.center,
                padding=ft.padding.only(16, 0, 16, 0),
                border=ft.border.only(bottom=ft.BorderSide(3, "transparent")),
                content=ft.Text(name, color=TAB_INACTIVE, size=14, weight=W600),
                on_click=lambda e, ix=i: self._show_tab(ix),
            )
            self.tab_bars.append(t)
            tabs.append(t)
        return ft.Container(
            height=56, bgcolor=NAVY,
            padding=ft.padding.only(left=20, right=12),
            content=ft.Row([
                ft.Text("Kapı → CAD Arama", color="white", size=17, weight=W600),
                ft.Text("fotoğraftan en benzer çizimi bulur",
                        color=TAB_INACTIVE, size=12.5),
                ft.Container(expand=True),
                ft.Row(tabs, spacing=2),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=14))

    def _show_tab(self, ix):
        for i, t in enumerate(self.tab_bars):
            t.border = ft.border.only(
                bottom=ft.BorderSide(3, AMBER if i == ix else "transparent"))
            t.content.color = "white" if i == ix else TAB_INACTIVE
        self.content.content = (self.search_view, self.add_view,
                                self.recent_view)[ix]
        if ix == 2:
            self._recent_refresh()
        self.page.update()

    # ====================================================== ARAMA EKRANI
    def _build_search_view(self):
        self.drop_inner = ft.Container(
            content=ft.Stack([
                ft.Image(src_base64=dashed_zone_b64(), width=268, height=190,
                         fit=ft.ImageFit.FILL),
                ft.Container(
                    width=268, height=190, alignment=ft.alignment.center,
                    content=ft.Column([
                        ft.Icon(ft.Icons.ADD_PHOTO_ALTERNATE_OUTLINED,
                                size=44, color=NAVY),
                        ft.Text("Kapı fotoğrafı seçmek için tıkla",
                                size=12.5, color=MUTED,
                                text_align=ft.TextAlign.CENTER),
                    ], spacing=8, alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER)),
            ]),
            on_click=lambda e: self.photo_picker.pick_files(
                dialog_title="Kapı fotoğrafı seçin", allow_multiple=False,
                allowed_extensions=PHOTO_EXTS),
            border_radius=14, ink=True,
        )
        self.photo_name = ft.Text("", size=11.5, color=MUTED, max_lines=1,
                                  overflow=ft.TextOverflow.ELLIPSIS)

        # açılır-kapanır "İşlenmiş hali"
        self.lineart_open = False
        self.lineart_chev = ft.Icon(ft.Icons.EXPAND_MORE, size=18, color=MUTED)
        self.lineart_img = ft.Container(
            width=268, height=170, bgcolor=BG, border_radius=10,
            alignment=ft.alignment.center,
            content=ft.Text("aramadan sonra burada\nçizgi hali görünür",
                            size=11.5, color=MUTED,
                            text_align=ft.TextAlign.CENTER))
        self.lineart_body = ft.Container(
            visible=False, alignment=ft.alignment.center,
            padding=ft.padding.only(top=6), content=self.lineart_img)
        lineart_header = ft.Container(
            content=ft.Row([ft.Text("İşlenmiş hali", size=13, weight=W600,
                                    color=TEXT),
                            ft.Container(expand=True), self.lineart_chev]),
            padding=ft.padding.symmetric(8, 10), border_radius=8,
            bgcolor=BG, on_click=self._toggle_lineart, ink=True)

        self.k_dd = ft.Dropdown(
            label="Sonuç sayısı", value="20", dense=True,
            options=[ft.dropdown.Option(v) for v in ("20", "30", "50")],
            border_color=BORDER, focused_border_color=NAVY, width=268,
            text_size=13, content_padding=ft.padding.symmetric(6, 12),
            on_change=lambda e: setattr(self, "k", int(e.control.value)))
        self.group_cb = ft.Checkbox(
            label="Benzer çizimleri grupla", value=True,
            active_color=NAVY, label_style=ft.TextStyle(size=13, color=TEXT),
            on_change=lambda e: setattr(self, "group", e.control.value))

        self.search_btn = ft.FilledButton(
            "Ara", icon=ft.Icons.SEARCH, disabled=True, height=44, expand=True,
            style=ft.ButtonStyle(
                bgcolor={"": NAVY, "disabled": "#B9C4D2"},
                color={"": "white", "disabled": "#F0F0F0"},
                shape=ft.RoundedRectangleBorder(radius=8),
                text_style=ft.TextStyle(size=14, weight=W600)),
            on_click=self._start_search)

        left = ft.Container(
            width=300, bgcolor=CARD, padding=16,
            border=ft.border.only(right=ft.BorderSide(1, BORDER)),
            content=ft.Column([
                ft.Text("Fotoğraf", size=13, weight=W600, color=TEXT),
                self.drop_inner,
                self.photo_name,
                ft.Container(height=6),
                lineart_header,
                self.lineart_body,
                ft.Container(height=10),
                self.k_dd,
                self.group_cb,
                ft.Container(expand=True),
                ft.Row([self.search_btn]),
            ], spacing=8))

        self.results_host = ft.Container(expand=True, bgcolor=BG,
                                         padding=ft.padding.all(16))
        self._show_empty_state()
        return ft.Row([left, self.results_host], spacing=0, expand=True)

    def _toggle_lineart(self, e):
        self.lineart_open = not self.lineart_open
        self.lineart_body.visible = self.lineart_open
        self.lineart_chev.name = (ft.Icons.EXPAND_LESS if self.lineart_open
                                  else ft.Icons.EXPAND_MORE)
        self.page.update()

    # ------------------------------------------------------------ foto seçimi
    def _photo_picked(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return
        self._set_photo(Path(e.files[0].path))

    def _set_photo(self, p: Path):
        self.photo_path = p
        b64 = b64_image(p, 540)
        self.drop_inner.content = ft.Container(
            width=268, height=190, border_radius=12, bgcolor=BG,
            border=ft.border.all(1.5, NAVY), alignment=ft.alignment.center,
            content=ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN,
                             border_radius=10) if b64 else
            ft.Text("(görsel açılamadı)", color=MUTED, size=12))
        self.photo_name.value = f"{p.name}   (değiştirmek için tıkla)"
        self.lineart_img.content = ft.Text(
            "aramadan sonra burada\nçizgi hali görünür", size=11.5,
            color=MUTED, text_align=ft.TextAlign.CENTER)
        self.search_btn.disabled = False
        self.page.update()

    # ------------------------------------------------------------- arama akışı
    def _start_search(self, e=None):
        if not self.photo_path:
            return
        self.search_btn.disabled = True
        self._show_loading("Model yükleniyor… (ilk aramada birkaç dakika sürebilir)"
                           if self.engine is None else "Aranıyor…")
        threading.Thread(target=self._search_worker,
                         args=(self.photo_path, self.k, self.group),
                         daemon=True).start()

    def _search_worker(self, photo: Path, k: int, group: bool):
        try:
            engine = self._get_engine()
            self._set_loading_text("Fotoğraf işleniyor… (kırpma + yazı silme + çizgi)")
            lineart, _, _ = engine.prepare_query(photo)
            self.lineart_img.content = ft.Image(
                src_base64=pil_to_b64(lineart), fit=ft.ImageFit.CONTAIN)
            self._set_loading_text("Katalog taranıyor…")
            if group:
                raw = engine.search_prepared(
                    lineart, k=min(k * 3, core.DUP_FETCH_CAP))
                results = core.group_results(raw, k)
            else:
                results = [(n, s, [])
                           for n, s in engine.search_prepared(lineart, k=k)]
            self._show_results(results)
        except Exception:
            traceback.print_exc()
            self._show_empty_state()
            self._snack("Arama başarısız — ayrıntı konsolda/logda.", DANGER)
        finally:
            self.search_btn.disabled = False
            self.page.update()

    # ------------------------------------------------------------ sağ bölge halleri
    def _show_empty_state(self):
        self.results_host.content = ft.Container(
            alignment=ft.alignment.center,
            content=ft.Column([
                ft.Icon(ft.Icons.IMAGE_SEARCH, size=88,
                        color=ft.Colors.with_opacity(0.18, NAVY)),
                ft.Text("Soldan bir kapı fotoğrafı seç, 'Ara' ile kataloğu tara",
                        size=14.5, color=MUTED),
                ft.Text("Sonuçlar burada kart kart listelenir; karta tıklayınca "
                        "adını kopyalayabilirsin", size=12, color="#9AA3AF"),
            ], spacing=10, alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    def _set_loading_text(self, msg):
        self.loading_text.value = msg
        self.page.update()

    def _show_loading(self, msg):
        self.loading_text = ft.Text(msg, size=13, color=MUTED)
        skels = ft.GridView(expand=True, max_extent=216, spacing=16,
                            run_spacing=16, child_aspect_ratio=0.66)
        for _ in range(10):
            skels.controls.append(ft.Container(
                bgcolor=CARD, border_radius=12, shadow=SHADOW,
                content=ft.Column([
                    ft.Container(expand=True, bgcolor="#EDF0F4",
                                 border_radius=ft.border_radius.only(12, 12, 0, 0)),
                    ft.Container(height=12, width=120, bgcolor="#EDF0F4",
                                 border_radius=6,
                                 margin=ft.margin.symmetric(10, 12)),
                ], spacing=0)))
        self.results_host.content = ft.Column([
            ft.Row([self.loading_text, ft.Container(width=10),
                    ft.Container(ft.ProgressBar(color=NAVY, bgcolor=NAVY_SOFT,
                                                height=4, border_radius=4),
                                 expand=True)]),
            ft.Container(height=10),
            skels,
        ], expand=True)
        self.page.update()

    def _show_results(self, results):
        self.selected_card = None
        grid = ft.GridView(expand=True, max_extent=216, spacing=16,
                           run_spacing=16, child_aspect_ratio=0.66)
        for i, (name, score, variants) in enumerate(results):
            grid.controls.append(self._result_card(i, name, score, variants))
        n_var = sum(len(v) for _, _, v in results)
        info = f"{len(results)} sonuç" + (f"  ·  +{n_var} benzer varyant gruplandı"
                                          if n_var else "")
        self.results_host.content = ft.Column([
            ft.Text(info, size=12.5, color=MUTED),
            ft.Container(height=6),
            grid,
        ], expand=True)

    # -------------------------------------------------------------- sonuç kartı
    def _score_badge(self, score):
        if score >= 0.75:
            bg, fg = AMBER, "#1A1A1A"
        elif score >= 0.55:
            bg, fg = NAVY_SOFT, NAVY
        else:
            bg, fg = "#E5E7EB", MUTED
        return ft.Container(
            content=ft.Text(f"{score:.3f}", size=11.5, weight=W600, color=fg),
            bgcolor=bg, border_radius=20,
            padding=ft.padding.symmetric(3, 9))

    def _result_card(self, rank, name, score, variants):
        img_b64 = b64_image(core.cad_image_path(name), 460)
        img = (ft.Image(src_base64=img_b64, fit=ft.ImageFit.CONTAIN)
               if img_b64 else ft.Text("(görsel açılamadı)", color=MUTED, size=12))

        overlays = [ft.Container(content=img, alignment=ft.alignment.center,
                                 padding=10, left=0, right=0, top=0, bottom=0)]
        if rank < 3:
            overlays.append(ft.Container(
                ft.Icon(ft.Icons.STAR_ROUNDED, size=18, color=AMBER),
                left=8, top=6))
        overlays.append(ft.Container(self._score_badge(score), right=8, top=8))
        if variants:
            overlays.append(ft.Container(
                content=ft.Text(f"+{len(variants)} varyant", size=10.5,
                                weight=W600, color=NAVY),
                bgcolor=NAVY_SOFT, border_radius=20,
                padding=ft.padding.symmetric(2, 8),
                right=8, bottom=8,
                on_click=lambda e, n=name, v=variants: self._open_variants(n, v)))

        strip = ft.Container(
            visible=False, left=0, right=0, bottom=0,
            bgcolor=ft.Colors.with_opacity(0.96, CARD),
            border=ft.border.only(top=ft.BorderSide(1, BORDER)),
            padding=ft.padding.symmetric(2, 4),
            content=ft.Row([
                ft.TextButton("Adı kopyala", icon=ft.Icons.CONTENT_COPY,
                              style=ft.ButtonStyle(color=NAVY,
                                                   text_style=ft.TextStyle(size=11.5)),
                              on_click=lambda e, n=name: self._copy_name(n)),
                ft.TextButton("Klasörde göster", icon=ft.Icons.FOLDER_OPEN,
                              style=ft.ButtonStyle(color=NAVY,
                                                   text_style=ft.TextStyle(size=11.5)),
                              on_click=lambda e, n=name: self._show_in_folder(n)),
            ], alignment=ft.MainAxisAlignment.SPACE_EVENLY, spacing=0))
        overlays.append(strip)

        title = ft.Container(
            content=ft.Text(f"{rank + 1}.  {name}", size=12, weight=W600,
                            color=TEXT, max_lines=1,
                            overflow=ft.TextOverflow.ELLIPSIS, tooltip=name),
            padding=ft.padding.only(10, 8, 10, 10))

        card = ft.Container(
            bgcolor=CARD, border_radius=12, shadow=SHADOW,
            border=ft.border.all(1.5, "transparent"),
            animate_scale=ft.Animation(120, ft.AnimationCurve.EASE_OUT),
            content=ft.Column([
                ft.Container(ft.Stack(overlays), expand=True, bgcolor=CARD,
                             border_radius=ft.border_radius.only(12, 12, 0, 0)),
                title,
            ], spacing=0))
        card._strip = strip
        card.on_hover = lambda e, c=card: self._card_hover(c, e.data == "true")
        card.on_click = lambda e, c=card: self._card_click(c)
        return card

    def _card_hover(self, card, on):
        if card is not self.selected_card:
            card.shadow = SHADOW_HOVER if on else SHADOW
            card.scale = 1.02 if on else 1.0
            card.update()

    def _card_click(self, card):
        if self.selected_card is card:
            card.border = ft.border.all(1.5, "transparent")
            card._strip.visible = False
            self.selected_card = None
        else:
            if self.selected_card is not None:
                self.selected_card.border = ft.border.all(1.5, "transparent")
                self.selected_card._strip.visible = False
                self.selected_card.update()
            card.border = ft.border.all(1.5, NAVY)
            card._strip.visible = True
            card.scale = 1.0
            self.selected_card = card
        card.update()

    def _open_variants(self, rep, variants):
        cells = []
        for name, score in variants:
            b64 = b64_image(core.cad_image_path(name), 380)
            cells.append(ft.Container(
                bgcolor=CARD, border_radius=12, border=ft.border.all(1, BORDER),
                padding=8, width=190,
                content=ft.Column([
                    ft.Container(
                        ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN)
                        if b64 else ft.Text("(görsel yok)", color=MUTED),
                        height=180, alignment=ft.alignment.center),
                    ft.Row([self._score_badge(score)],
                           alignment=ft.MainAxisAlignment.CENTER),
                    ft.Text(name, size=11, color=TEXT, max_lines=1,
                            overflow=ft.TextOverflow.ELLIPSIS, tooltip=name),
                    ft.TextButton("Adı kopyala", icon=ft.Icons.CONTENT_COPY,
                                  style=ft.ButtonStyle(
                                      color=NAVY,
                                      text_style=ft.TextStyle(size=11)),
                                  on_click=lambda e, n=name: self._copy_name(n)),
                ], spacing=6,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER)))
        dlg = ft.AlertDialog(
            title=ft.Text(f"{rep} — {len(variants)} benzer varyant",
                          size=15, weight=W600),
            bgcolor=BG,
            content=ft.Container(
                width=680,
                content=ft.Column([ft.Row(cells, wrap=True, spacing=12,
                                          run_spacing=12)],
                                  scroll=ft.ScrollMode.AUTO, height=460)),
            actions=[ft.TextButton("Kapat",
                                   on_click=lambda e: self.page.close(dlg))])
        self.page.open(dlg)

    # ====================================================== KAPI EKLE EKRANI
    def _build_add_view(self):
        info = ft.Text(
            "Yeni kapıyı kataloğa ekle: solda kapının FOTOĞRAFLARINI, sağda "
            "ÇİZİMLERİNİ seç. Çizim olarak doğrudan AutoCAD dosyası (DWG/DXF) "
            "verebilirsin — PNG'ye otomatik çevrilir; PNG/JPG export da olur. "
            "Fotoğraf ve çizim İKİSİ DE zorunludur, kapı katalogda çizim "
            "dosyasının adıyla görünür. Tek seferde TEK kapı eklenir (aynı "
            "kapının birden çok fotoğrafı ve çizim varyantı olabilir).",
            size=12.5, color=MUTED)

        self.add_panels = {}
        cols = []
        for key, title, icon, picker in (
                ("foto", "Kapı Fotoğrafları", ft.Icons.PHOTO_CAMERA_OUTLINED,
                 self.add_foto_picker),
                ("cizim", "CAD Çizimleri", ft.Icons.ARCHITECTURE,
                 self.add_cizim_picker)):
            zone = ft.Container(
                content=ft.Stack([
                    ft.Image(src_base64=dashed_zone_b64(430, 110, 12),
                             width=430, height=110, fit=ft.ImageFit.FILL),
                    ft.Container(
                        width=430, height=110, alignment=ft.alignment.center,
                        content=ft.Column([
                            ft.Icon(icon, size=28, color=NAVY),
                            ft.Text("Dosya seçmek için tıkla"
                                    + ("  (DWG/DXF/PNG/JPG)" if key == "cizim"
                                       else "  (PNG/JPG)"),
                                    size=11.5, color=MUTED),
                        ], spacing=4,
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER)),
                ]),
                on_click=lambda e, k=key, p=picker: p.pick_files(
                    dialog_title=("Kapı fotoğraflarını seçin" if k == "foto" else
                                  "CAD çizimlerini seçin (DWG/DXF ya da PNG/JPG)"),
                    allow_multiple=True,
                    allowed_extensions=(PHOTO_EXTS if k == "foto"
                                        else DRAWING_EXTS)),
                border_radius=12, ink=True)
            files_row = ft.Row(wrap=True, spacing=10, run_spacing=10)
            self.add_panels[key] = files_row
            cols.append(ft.Container(
                expand=True, bgcolor=CARD, border_radius=12, shadow=SHADOW,
                padding=16,
                content=ft.Column([
                    ft.Text(title, size=14, weight=W600, color=NAVY),
                    ft.Row([zone], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Container(ft.Column([files_row],
                                           scroll=ft.ScrollMode.AUTO),
                                 expand=True, padding=ft.padding.only(top=10)),
                ], spacing=10)))

        self.add_status = ft.Text("Hazır.", size=12.5, color=MUTED)
        self.add_progress = ft.ProgressBar(color=NAVY, bgcolor=NAVY_SOFT,
                                           height=4, border_radius=4,
                                           visible=False)
        self.add_btn = ft.FilledButton(
            "Kataloğa Ekle", icon=ft.Icons.CHECK, height=44, expand=True,
            style=ft.ButtonStyle(
                bgcolor={"": NAVY, "disabled": "#B9C4D2"},
                color={"": "white", "disabled": "#F0F0F0"},
                shape=ft.RoundedRectangleBorder(radius=8),
                text_style=ft.TextStyle(size=14, weight=W600)),
            on_click=self._start_add)

        return ft.Container(
            padding=16, expand=True,
            content=ft.Column([
                ft.Container(info, bgcolor=CARD, border_radius=12,
                             shadow=SHADOW, padding=ft.padding.symmetric(10, 14)),
                ft.Row(cols, spacing=16, expand=True,
                       vertical_alignment=ft.CrossAxisAlignment.STRETCH),
                self.add_progress,
                ft.Row([self.add_btn]),
                ft.Row([self.add_status,
                        ft.Container(expand=True),
                        ft.TextButton("Listeyi temizle",
                                      style=ft.ButtonStyle(color=MUTED),
                                      on_click=self._add_clear)]),
            ], spacing=10))

    def _add_picked(self, e: ft.FilePickerResultEvent, key):
        if not e.files:
            return
        target = self.add_photos if key == "foto" else self.add_drawings
        for f in e.files:
            if f.path not in target:
                target.append(f.path)
        self._add_refresh(key)

    def _add_refresh(self, key):
        row = self.add_panels[key]
        target = self.add_photos if key == "foto" else self.add_drawings
        row.controls.clear()
        for p in list(target):
            pp = Path(p)
            if pp.suffix.lower() in (".dwg", ".dxf"):
                gorsel = ft.Container(
                    width=120, height=110, bgcolor=NAVY_SOFT, border_radius=8,
                    alignment=ft.alignment.center,
                    content=ft.Column([
                        ft.Icon(ft.Icons.ARCHITECTURE, size=30, color=NAVY),
                        ft.Text("AutoCAD çizimi\n(PNG'ye çevrilir)", size=10,
                                color=MUTED, text_align=ft.TextAlign.CENTER),
                    ], spacing=4, alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER))
            else:
                b64 = b64_image(pp, 240)
                gorsel = ft.Container(
                    width=120, height=110, bgcolor=BG, border_radius=8,
                    alignment=ft.alignment.center,
                    content=ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN)
                    if b64 else ft.Text("(açılamadı)", size=10, color=MUTED))
            row.controls.append(ft.Container(
                bgcolor=CARD, border_radius=10, border=ft.border.all(1, BORDER),
                padding=6,
                content=ft.Column([
                    gorsel,
                    ft.Text(pp.name, size=10.5, color=TEXT, width=120,
                            max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                            tooltip=pp.name),
                    ft.TextButton("✕ çıkar",
                                  style=ft.ButtonStyle(
                                      color=DANGER,
                                      text_style=ft.TextStyle(size=10.5)),
                                  on_click=lambda e, path=p, k=key:
                                  self._add_remove(path, k)),
                ], spacing=2,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER)))
        self.add_status.value = (f"{len(self.add_photos)} fotoğraf, "
                                 f"{len(self.add_drawings)} çizim seçili.")
        self.add_status.color = MUTED
        self.page.update()

    def _add_remove(self, path, key):
        target = self.add_photos if key == "foto" else self.add_drawings
        if path in target:
            target.remove(path)
        self._add_refresh(key)

    def _add_clear(self, e=None):
        self.add_photos.clear()
        self.add_drawings.clear()
        self._add_refresh("foto")
        self._add_refresh("cizim")
        self.add_status.value = "Liste temizlendi."
        self.page.update()

    def _start_add(self, e=None):
        if not self.add_photos or not self.add_drawings:
            self.add_status.value = ("Eksik: en az bir FOTOĞRAF ve bir ÇİZİM "
                                     f"seçmelisin (şu an {len(self.add_photos)} "
                                     f"foto, {len(self.add_drawings)} çizim).")
            self.add_status.color = DANGER
            self.page.update()
            return
        self.add_btn.disabled = True
        self.add_progress.visible = True
        self.add_status.value = "Ekleniyor… (ilk eklemede model yüklemesi sürebilir)"
        self.add_status.color = MUTED
        self.page.update()
        threading.Thread(target=self._add_worker,
                         args=(list(self.add_drawings), list(self.add_photos)),
                         daemon=True).start()

    def _add_worker(self, drawings, photos):
        import katalog
        try:
            def log(msg):
                self.add_status.value = str(msg)
                self.page.update()
            engine = self._get_engine()
            msg = katalog.add_entry(drawings, photos, engine=engine, log=log)
            core._base_emb = None      # gruplama önbelleği yenilensin
            self.add_photos.clear()
            self.add_drawings.clear()
            self._add_refresh("foto")
            self._add_refresh("cizim")
            self.add_status.value = "✔ " + str(msg)
            self.add_status.color = OK_GREEN
            self._snack("Kapı kataloğa eklendi.", OK_GREEN)
        except Exception as ex:
            traceback.print_exc()
            self.add_status.value = f"HATA: {ex}"
            self.add_status.color = DANGER
            self._snack("Ekleme başarısız — ayrıntı durum satırında.", DANGER)
        finally:
            self.add_btn.disabled = False
            self.add_progress.visible = False
            self.page.update()

    # ================================================== SON EKLENENLER EKRANI
    def _build_recent_view(self):
        self.recent_status = ft.Text("", size=12.5, color=NAVY)
        self.recent_list = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True,
                                     spacing=10)
        return ft.Container(
            padding=16, expand=True,
            content=ft.Column([
                ft.Container(
                    bgcolor=CARD, border_radius=12, shadow=SHADOW,
                    padding=ft.padding.symmetric(10, 14),
                    content=ft.Row([
                        ft.Text("Uygulamadan eklenen kapılar (yeniden eskiye). "
                                "Yanlış eklediysen 'Sil' ile hem çizimleri hem "
                                "fotoğrafları hem de eşleşme kayıtlarını tamamen "
                                "geri alırsın. Orijinal katalog buradan silinemez.",
                                size=12.5, color=MUTED, expand=True),
                        self.recent_status])),
                self.recent_list,
            ], spacing=10))

    def _recent_refresh(self):
        import katalog
        self.recent_list.controls.clear()
        try:
            entries = katalog.list_entries()
        except Exception:
            traceback.print_exc()
            entries = []
        if not entries:
            self.recent_status.value = ""
            self.recent_list.controls.append(ft.Container(
                alignment=ft.alignment.center, padding=40,
                content=ft.Column([
                    ft.Icon(ft.Icons.INBOX_OUTLINED, size=64,
                            color=ft.Colors.with_opacity(0.2, NAVY)),
                    ft.Text("Henüz uygulamadan kapı eklenmedi.",
                            size=13.5, color=MUTED),
                ], spacing=8,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER)))
            return
        self.recent_status.value = f"{len(entries)} ekleme"
        import katalog as kt
        for entry in entries:
            thumbs = ft.Row(spacing=6)
            for ph in entry.get("fotolar", [])[:4]:
                b64 = b64_image(kt.ADDED_PHOTOS_DIR / ph, 220)
                thumbs.controls.append(ft.Container(
                    width=96, height=96, bgcolor=BG, border_radius=8,
                    alignment=ft.alignment.center,
                    content=ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN,
                                     border_radius=8)
                    if b64 else ft.Text("(foto yok)", size=10, color=MUTED)))
            cizimler = entry.get("cizimler", [])
            fotolar = entry.get("fotolar", [])
            cizim_rows = ft.Column(spacing=2)
            for cz in cizimler:
                cizim_rows.controls.append(ft.Row([
                    ft.Text("• " + cz, size=12, color=TEXT),
                    ft.IconButton(ft.Icons.CLOSE, icon_size=14,
                                  icon_color=DANGER,
                                  tooltip="Bu çizimi katalogdan sil",
                                  style=ft.ButtonStyle(
                                      padding=ft.padding.all(2)),
                                  on_click=lambda e, en=entry, c=cz:
                                  self._recent_delete_drawing_ask(en, c)),
                ], spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER))
            self.recent_list.controls.append(ft.Container(
                bgcolor=CARD, border_radius=12, shadow=SHADOW, padding=14,
                content=ft.Row([
                    thumbs,
                    ft.Column([
                        ft.Text(f"{entry.get('tarih', '?')}  —  "
                                f"{len(fotolar)} fotoğraf, {len(cizimler)} "
                                "çizim eklendi", size=13, weight=W600,
                                color=TEXT),
                        cizim_rows,
                    ], spacing=6, expand=True),
                    ft.OutlinedButton(
                        "Sil", icon=ft.Icons.DELETE_OUTLINE,
                        style=ft.ButtonStyle(
                            color=DANGER,
                            side=ft.BorderSide(1.2, DANGER),
                            shape=ft.RoundedRectangleBorder(radius=8)),
                        on_click=lambda e, en=entry: self._recent_delete_ask(en)),
                ], spacing=14,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER)))

    def _confirm(self, title, msg, on_yes):
        def yes(e):
            self.page.close(dlg)
            on_yes()
        dlg = ft.AlertDialog(
            title=ft.Text(title, size=15, weight=W600),
            content=ft.Text(msg, size=13),
            actions=[
                ft.TextButton("Vazgeç", on_click=lambda e: self.page.close(dlg)),
                ft.FilledButton("Sil", style=ft.ButtonStyle(bgcolor=DANGER,
                                                            color="white"),
                                on_click=yes),
            ])
        self.page.open(dlg)

    def _recent_delete_ask(self, entry):
        cizimler = ", ".join(entry.get("cizimler", []))
        self._confirm(
            "Silme onayı",
            f"Bu ekleme tamamen geri alınacak:\n\n{cizimler}\n\nÇizimler arama "
            "kataloğundan, fotoğraflar ve eşleşme kayıtları diskten silinecek. "
            "Emin misin?",
            lambda: threading.Thread(target=self._recent_delete_worker,
                                     args=(entry,), daemon=True).start())

    def _recent_delete_worker(self, entry):
        import katalog
        self.recent_status.value = "Siliniyor…"
        self.page.update()
        try:
            msg = katalog.remove_entry(entry, engine=self.engine, log=print)
            core._base_emb = None
            self._snack(str(msg), OK_GREEN)
        except Exception as ex:
            traceback.print_exc()
            self._snack(f"HATA: {ex}", DANGER)
        self._recent_refresh()
        self.page.update()

    def _recent_delete_drawing_ask(self, entry, drawing_name):
        kalan = len(entry.get("cizimler", [])) - 1
        ek = ("\n\nBu kayıttaki SON çizim: fotoğraflarıyla birlikte ekleme "
              "komple geri alınacak." if kalan == 0 else
              f"\n\nKayıttaki diğer {kalan} çizim ve fotoğraflar kalacak.")
        self._confirm(
            "Çizim silme onayı",
            f"'{drawing_name}' katalogdan silinecek.{ek}\nEmin misin?",
            lambda: threading.Thread(target=self._recent_delete_drawing_worker,
                                     args=(entry, drawing_name),
                                     daemon=True).start())

    def _recent_delete_drawing_worker(self, entry, drawing_name):
        import katalog
        self.recent_status.value = "Siliniyor…"
        self.page.update()
        try:
            msg = katalog.remove_drawing(entry, drawing_name,
                                         engine=self.engine, log=print)
            core._base_emb = None
            self._snack(str(msg), OK_GREEN)
        except Exception as ex:
            traceback.print_exc()
            self._snack(f"HATA: {ex}", DANGER)
        self._recent_refresh()
        self.page.update()

    # --------------------------------------------------------------- eylemler
    def _snack(self, msg, color=NAVY):
        self.page.open(ft.SnackBar(ft.Text(msg, color="white"),
                                   bgcolor=color, duration=2500))

    def _copy_name(self, name):
        self.page.set_clipboard(name)
        self._snack(f"Panoya kopyalandı: {name}")

    def _show_in_folder(self, name):
        p = core.cad_image_path(name)
        if p.exists():
            subprocess.Popen(["explorer", "/select,", str(p)])
        else:
            self._snack("Dosya bulunamadı", DANGER)

    # ------------------------------------------------------------------- demo
    def _demo_flow(self):
        time.sleep(1.5)
        photo = next(p for pat in ("*.jpg", "*.jpeg", "*.png")
                     for p in sorted((ROOT / "photos").glob(pat)))
        self._set_photo(photo)
        time.sleep(0.4)
        self._start_search()


def main(page: ft.Page):
    FletApp(page)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        arg = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
        sys.exit(core.selftest(arg))
    ft.app(target=main)
