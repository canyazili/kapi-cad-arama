# Kapı → CAD Arama

Kapı **fotoğrafından**, 32.000+ AutoCAD çiziminin içinden en benzerini bulan görsel arama sistemi.
Masaüstü uygulaması (Tkinter) + eğitim/değerlendirme scriptleri.

**Güncel başarı** (445 hiç görülmemiş test fotoğrafı, gruplu kart görünümü):
ilk kartta doğru tasarım **%42,7** · ilk 20 kartta **%77,5**.

## Nasıl çalışıyor?

```
fotoğraf ──► kapı-crop ──► OCR ile metin silme ──► HED lineart
                                                      │
                     DINOv3 embedding ◄───────────────┤
                     DINOv2 embedding ◄───────────────┘
                            │  (her omurganın kendi projeksiyon MLP'si)
                            ▼
              skorların ağırlıklı toplamı (0.5/0.5 ensemble)
                            ▼
              FAISS benzerlik sıralaması ──► ikiz-tekilleştirme ──► sonuç kartları
```

- **Omurgalar:** donuk DINOv2 vitb14 + DINOv3 vitb16; ikisi farklı hatalar yaptığı için
  ensemble tek başına en iyiden ~6 puan fazla verir.
- **Projeksiyon:** her omurga üstünde 768→512 MLP; InfoNCE ile eğitilir
  (sentetik clean↔HED çiftleri + insan-onaylı foto↔CAD etiketleri).
  Batch içi negatiflerde **ikiz maskesi** (base-benzerlik ≥0.95 olan katalog kopyaları
  negatif sayılmaz).
- **İkiz-tekilleştirme:** katalogdaki çizimlerin %84'ünün yakın kopyası var; sonuçlar
  ham DINOv2 uzayında 0.95 eşiğiyle gruplanır, her kart bir tasarımı temsil eder.

## Depo düzeni

```
search.py               arama çekirdeği (embedder'lar, füzyon, FAISS)
kapi_arama_app.py       masaüstü uygulaması (+ --selftest modu)
configs/config.yaml     tüm ayarlar (aktif indeks, füzyon ağırlıkları, yollar)
scripts/
  01_clean_cad.py         CAD PNG temizliği
  02_photo_to_lineart.py  foto → lineart boru hattı (crop + OCR + HED)
  03_build_index.py       embedding + FAISS indeks kurulumu
  04_evaluate.py          recall@k değerlendirmesi (labels_clean ile)
  05..09_*                teşhis / etiket onay araçları
  07_train_projection.py  projeksiyon eğitimi (--backbone dinov2|dinov3, --final,
                          --twin-mask-sim, --out-variant)
  10_assist_label.py      model destekli etiketleme sayfası (--top-k, --group)
  experiments/            tek seferlik deney scriptleri (kıyas, ensemble taraması)
app/main.py             eski Streamlit arayüzü
dist/kurulum.iss        Inno Setup kurulum betiği (tek dosyalık dağıtım)
KapiArama.spec          PyInstaller exe tarifi
```

**Repoda OLMAYAN** (boyut nedeniyle .gitignore'da): `cad_png/`, `cad_dwg/`, `photos/`
(kaynak veriler), `index/` ve `data/` altındaki üretilmiş embedding/indeksler, `.venv`, `dist/paket`.
Repoya giren değerli küçük dosyalar: `data/eval/labels_clean.json` (insan-onaylı etiketler),
`data/eval/projection_split.json` (donmuş eğitim/val/test ayrımı — kıyaslanabilirlik için kritik),
`index/excluded_trivial.json` (parça-profil dışlama onayları), `labels.json` (ilk etiketler).

## Kurulum (geliştirme)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# DİKKAT: torch'u CUDA ile kurun, düz "pip install torch" CPU sürümüyle ezebilir:
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

- DINOv3 ağırlıkları HuggingFace'te **gated**: hesapla
  [model sayfasında](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
  erişim isteyin, `huggingface-cli login` ile token girin.
- Veri klasörlerini (cad_png, photos) yerleştirin, sonra sırayla:
  `01_clean_cad` → `03_build_index` → (etiketler varsa) `07_train_projection`.

## Önemli kurallar

- `cad_png/` ve `photos/` **salt okunur**; orijinal `labels.json`'a asla dokunulmaz.
- Deney kıyasları HER ZAMAN donmuş ayrımlı `projected` varyantı + test kümesiyle yapılır;
  `projected_final*` (tüm veriyle eğitilen üretim modelleri) dürüstçe ölçülemez.
- `faiss.write_index/read_index` Windows'ta Türkçe karakterli yollarda çalışmaz;
  serialize/deserialize + Python G/Ç kullanılır (kod zaten böyle).

## Dağıtım

- **Taşınabilir ZIP:** `dist/paket` klasörü (exe + modeller + indeksler, internet gerektirmez)
  arşivlenip dağıtılır. Paket güncelleme: yeni `index/variants/projected_final*` klasörlerini
  ve `configs/config.yaml`'ı `dist/paket`'e kopyala; kod değiştiyse PyInstaller ile exe'yi
  yeniden derle (`pyinstaller KapiArama.spec --clean`; önce `TCL_LIBRARY`/`TK_LIBRARY`
  ortam değişkenlerini ayarla).
- **Tek dosyalık kurulum:** `ISCC.exe dist/kurulum.iss` (Inno Setup 6).
