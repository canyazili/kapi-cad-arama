# -*- coding: utf-8 -*-
"""cad_normalize.py — CAD görsellerini standart forma getiren çekirdek fonksiyon.

`normalize_cad` tek bir AutoCAD export görselini (PNG/JPG) beyaz zemin + koyu çizgi,
dikey, kare 518x518 forma getirir. Sadece numpy + PIL kullanır; hiçbir dosya/konfig
bağımlılığı yoktur.

Tek doğruluk kaynağı burasıdır:
  - scripts/01_clean_cad.py (toplu normalizasyon CLI'ı) buradan import eder,
  - katalog.py (uygulamadan "Kapı Ekle") buradan import eder.
Böylece paketlenmiş exe içinde scripts/ klasörüne ihtiyaç kalmaz.
"""
import numpy as np
from PIL import Image

# Koyu piksel eşiği: bbox tespiti bu değerin altındaki pikselleri "içerik" sayar
DARK_THRESHOLD = 200
# Zemin koyu mu kararı: ortalama parlaklık bunun altındaysa invert edilir
INVERT_MEAN_THRESHOLD = 128
# Crop sonrası içeriğin etrafına bırakılan pay (uzun kenarın oranı)
BBOX_MARGIN_RATIO = 0.02


def normalize_cad(img: Image.Image, target_size: int) -> Image.Image:
    """Tek bir CAD görselini standart forma getirir (beyaz zemin, koyu çizgi, dikey, kare)."""
    # 1) Şeffaflık gerçekten kullanılmışsa çizgi rengine hiç bakma, doğrudan
    #    alpha kanalını kullan: opak piksel (çizgi) koyu, şeffaf zemin beyaz olur.
    #    Böylece şeffaf zemin üzerine BEYAZ çizgili exportlar da doğru çıkar
    #    (beyaza composite edilince çizginin kaybolduğu hata bu yolla çözülür).
    arr = None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        if alpha.min() < 255:  # şeffaflık gerçekten var
            arr = 255 - alpha
        else:  # alpha tamamen opak: şeffaflık bilgisi yok, normal yoldan devam
            img = rgba

    if arr is None:
        # 2) Alpha yolu kullanılmadıysa: griye çevir, zemin koyuysa invert
        #    -> hedef: beyaz zemin üzerine koyu çizgi
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        if arr.mean() < INVERT_MEAN_THRESHOLD:
            arr = 255 - arr

    # 3) Koyu piksellere göre içerik bbox'ı
    if arr.min() == arr.max():
        raise ValueError("görsel tek renk — boş/bozuk export, yeniden export gerekli")
    mask = arr < DARK_THRESHOLD
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("içerik bulunamadı (görsel tamamen boş/beyaz)")
    margin = int(max(arr.shape) * BBOX_MARGIN_RATIO)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, arr.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, arr.shape[1])
    arr = arr[y0:y1, x0:x1]

    # 4) Yatay export edilmişse dikeye döndür
    h, w = arr.shape
    if w > h:
        arr = np.rot90(arr)  # 90 derece, oran korunur
        h, w = arr.shape

    # 5) Beyaz zeminle kare padding (oran bozulmaz), sonra resize
    side = max(h, w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = arr

    out = Image.fromarray(canvas).resize((target_size, target_size), Image.LANCZOS)
    return out.convert("RGB")


def make_debug_image(before: Image.Image, after: Image.Image) -> Image.Image:
    """Önce/sonra halini yan yana gösteren kontrol görseli üretir."""
    h = 400
    b = before.convert("RGB")
    b = b.resize((max(1, int(b.width * h / b.height)), h))
    a = after.convert("RGB").resize((h, h))
    canvas = Image.new("RGB", (b.width + a.width + 10, h), (255, 0, 0))
    canvas.paste(b, (0, 0))
    canvas.paste(a, (b.width + 10, 0))
    return canvas
