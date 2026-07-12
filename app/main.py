# -*- coding: utf-8 -*-
"""
app/main.py — Streamlit arama arayüzü.

Kullanıcı bir kapı görseli yükler; sorgunun crop edilmiş ve lineart hali
debug amaçlı yan panelde gösterilir, altta top-20 en benzer CAD çizimi
(dosya adı + benzerlik skoru) grid halinde listelenir.

Tüm arama mantığı search.py'dedir; burada sadece arayüz vardır.

Çalıştırma:
  streamlit run app/main.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search import SearchEngine, load_config  # noqa: E402

GRID_COLS = 4


@st.cache_resource(show_spinner="Model ve indeks yükleniyor...")
def get_cached_engine():
    return SearchEngine()


def main():
    st.set_page_config(page_title="Kapı CAD Arama", layout="wide")
    st.title("Kapı görselinden CAD çizimi arama")

    cfg = load_config()
    top_k = cfg["top_k"]
    cad_clean_dir = ROOT / cfg["paths"]["cad_clean"]
    cad_png_dir = ROOT / cfg["paths"]["cad_png"]

    try:
        engine = get_cached_engine()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    uploaded = st.file_uploader("Kapı görseli yükleyin",
                                type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Aramak için bir kapı fotoğrafı ya da katalog görseli yükleyin.")
        st.stop()

    query = Image.open(uploaded)
    query.load()

    with st.spinner("Sorgu işleniyor ve aranıyor..."):
        lineart, cropped, cleaned = engine.prepare_query(query)
        results = engine.search_prepared(lineart, k=top_k)

    st.subheader("Sorgu (debug)")
    steps = [(query, "Yüklenen görsel"), (cropped, "Kapı-crop")]
    if cleaned is not None:
        steps.append((cleaned, "Metin temizlenmiş"))
    steps.append((lineart, "HED lineart (arama girdisi)"))
    for col, (img, caption) in zip(st.columns(len(steps)), steps):
        col.image(img, caption=caption, use_container_width=True)

    st.subheader(f"En benzer {len(results)} CAD çizimi")
    for start in range(0, len(results), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for col, (name, score) in zip(cols, results[start:start + GRID_COLS]):
            img_path = cad_clean_dir / name
            if not img_path.exists():  # temiz hali silinmişse orijinali göster
                img_path = cad_png_dir / name
            with col:
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning("görsel bulunamadı")
                st.caption(f"**{name}**  \nskor: {score:.4f}")


if __name__ == "__main__":
    main()
