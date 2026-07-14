# -*- coding: utf-8 -*-
"""dwg2png.py — AutoCAD DWG/DXF dosyasını PNG görsele çevirir.

Akış: DWG ise önce ODA File Converter (ücretsiz, opendesign.com) ile DXF'e
çevrilir; DXF, ezdxf'in matplotlib backend'iyle beyaz zemin + siyah çizgi
olarak render edilir. Çıkan PNG, katalog ekleme hattındaki normalize_cad'e
verilecek formattadır.

ODA File Converter aranan yerler (sırayla):
  1) KAPI_ODA ortam değişkeni (exe'nin tam yolu)
  2) taşınabilir pakette exe'nin yanında araclar/ODAFileConverter*/
  3) C:/Program Files/ODA/ODAFileConverter*/
Bulunamazsa DWG için anlaşılır bir hata verilir (DXF yine çalışır).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RENDER_PX = 2000   # render edilen PNG'nin uzun kenarı (normalize_cad zaten küçültür)


def find_oda_converter():
    """ODAFileConverter.exe yolunu döner; yoksa None."""
    env = os.environ.get("KAPI_ODA")
    if env and Path(env).exists():
        return Path(env)
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent / "araclar")
    roots += [Path(r"C:/Program Files/ODA"), Path(r"C:/Program Files (x86)/ODA")]
    for root in roots:
        if root.exists():
            hits = sorted(root.glob("ODAFileConverter*/ODAFileConverter.exe"),
                          reverse=True)
            if hits:
                return hits[0]
    return None


def dwg_to_dxf(dwg_path: Path, out_dir: Path) -> Path:
    """ODA File Converter ile tek DWG'yi DXF'e çevirir; DXF yolunu döner."""
    oda = find_oda_converter()
    if oda is None:
        raise RuntimeError(
            "DWG çevirmek için ODA File Converter gerekli (ücretsiz):\n"
            "https://www.opendesign.com/guestfiles/oda_file_converter\n"
            "Kurulumdan sonra uygulamayı yeniden başlatın.")
    # ODA klasör bazlı çalışır: DWG'yi geçici giriş klasörüne kopyala
    in_dir = out_dir / "_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dwg_path, in_dir / dwg_path.name)
    # Argümanlar: girdi çıktı sürüm tür özyineleme denetim [filtre]
    cmd = [str(oda), str(in_dir), str(out_dir), "ACAD2018", "DXF", "0", "1",
           dwg_path.name]
    subprocess.run(cmd, check=True, timeout=120,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    dxf = out_dir / (dwg_path.stem + ".dxf")
    if not dxf.exists():
        # bazı sürümler farklı büyük/küçük harfle yazar
        cands = list(out_dir.glob(dwg_path.stem + ".*"))
        cands = [c for c in cands if c.suffix.lower() == ".dxf"]
        if not cands:
            raise RuntimeError(f"ODA çevirisi başarısız: {dwg_path.name} "
                               f"(çıktı klasöründe DXF yok)")
        dxf = cands[0]
    return dxf


def dxf_to_png(dxf_path: Path, out_png: Path):
    """DXF'i beyaz zemin + siyah çizgiyle PNG'ye render eder (ezdxf+matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")               # pencere açma, dosyaya çiz
    import matplotlib.pyplot as plt
    import ezdxf
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from ezdxf.addons.drawing.properties import LayoutProperties

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    fig = plt.figure(dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    props = LayoutProperties.from_layout(msp)
    props.set_colors("#FFFFFF")         # beyaz zemin -> varsayılan çizgiler siyah
    frontend_kw = {}
    try:  # katman renklerini (kırmızı/sarı ölçü çizgileri) hep siyaha zorla
        from ezdxf.addons.drawing.config import Configuration, ColorPolicy
        frontend_kw["config"] = Configuration(color_policy=ColorPolicy.BLACK)
    except Exception:
        pass                            # eski ezdxf: renkli kalır, normalize halleder
    Frontend(ctx, MatplotlibBackend(ax), **frontend_kw).draw_layout(
        msp, finalize=True, layout_properties=props)
    # uzun kenar RENDER_PX olacak şekilde boyutlandır
    w, h = fig.get_size_inches()
    scale = RENDER_PX / (max(w, h) * 100)
    fig.set_size_inches(w * scale, h * scale)
    fig.savefig(out_png, dpi=100, facecolor="#FFFFFF")
    plt.close(fig)


def cad_to_png(path, out_png=None) -> Path:
    """DWG/DXF -> PNG. PNG yolunu döner; out_png verilmezse geçici dosya."""
    path = Path(path)
    if out_png is None:
        out_png = Path(tempfile.mkdtemp(prefix="kapi_dwg_")) / (path.stem + ".png")
    out_png = Path(out_png)
    with tempfile.TemporaryDirectory(prefix="kapi_oda_") as td:
        if path.suffix.lower() == ".dwg":
            dxf = dwg_to_dxf(path, Path(td))
        elif path.suffix.lower() == ".dxf":
            dxf = path
        else:
            raise ValueError(f"DWG/DXF bekleniyordu: {path.name}")
        dxf_to_png(dxf, out_png)
    return out_png
