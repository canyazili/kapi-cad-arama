# -*- mode: python ; coding: utf-8 -*-
# Flet arayüzlü (kapi_arama_flet.py) üretim paketi.
# Eski KapiArama.spec Tkinter girişini kullanıyordu; bu spec flet + flet_desktop
# (Flutter istemcisi), DWG/DXF çevirme (ezdxf + matplotlib) ve lazy import edilen
# yerel modülleri (katalog, dwg2png) bundle'a katar. Tkinter dışlanır (flet yolu
# hiç kullanmaz; TCL/TK runtime derdini de ortadan kaldırır).
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['cv2', 'tqdm', 'yaml',
                 'core', 'search', 'katalog', 'dwg2png', 'cad_normalize',
                 'photo_lineart']
for pkg in ('torch', 'torchvision', 'controlnet_aux', 'easyocr', 'timm', 'faiss',
            'transformers', 'tokenizers', 'safetensors', 'huggingface_hub',
            'regex', 'flet', 'flet_desktop', 'ezdxf', 'matplotlib'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h


a = Analysis(
    ['kapi_arama_flet.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', '_tkinter', 'Tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KapiArama',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='marka/app_icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KapiArama',
)
