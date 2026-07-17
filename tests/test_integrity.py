# -*- coding: utf-8 -*-
"""Veri sızıntısı, indeks hizası ve taşınabilirlik için hızlı bütünlük testleri."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml
import faiss


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from label_tools import photo_root  # noqa: E402


class ProjectIntegrityTests(unittest.TestCase):
    def test_experiment_scripts_do_not_contain_machine_specific_root(self):
        offenders = []
        for path in (ROOT / "scripts" / "experiments").glob("*.py"):
            text = path.read_text(encoding="utf-8-sig").lower()
            if "users/canya/desktop" in text or "users\\canya\\desktop" in text:
                offenders.append(path.name)
        self.assertEqual([], offenders)

    def test_required_runtime_dependencies_are_declared(self):
        declared = {
            line.strip().lower().split("==", 1)[0]
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        required = {"flet", "transformers", "ezdxf", "matplotlib", "pyinstaller"}
        self.assertFalse(required - declared, f"Eksik bağımlılıklar: {required - declared}")

    def test_frozen_split_has_no_photo_or_family_leakage(self):
        path = ROOT / "data" / "eval" / "projection_split.json"
        if not path.exists():
            self.skipTest("projection_split.json yok")
        split = json.loads(path.read_text(encoding="utf-8"))
        names = {key: set(split[key]) for key in ("train", "val", "test")}
        families = {key: {photo_root(p) for p in value} for key, value in names.items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(names[left] & names[right], f"Fotoğraf sızıntısı: {left}/{right}")
            self.assertFalse(
                families[left] & families[right], f"Aile sızıntısı: {left}/{right}"
            )

    def test_frozen_split_entries_exist_in_clean_labels(self):
        split_path = ROOT / "data" / "eval" / "projection_split.json"
        labels_path = ROOT / "data" / "eval" / "labels_clean.json"
        if not split_path.exists() or not labels_path.exists():
            self.skipTest("Donmuş split veya temiz etiketler yok")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))["eslesme"]
        missing = {
            p for part in ("train", "val", "test") for p in split[part] if p not in labels
        }
        self.assertFalse(missing, f"Etiketi olmayan split fotoğrafları: {sorted(missing)[:5]}")

    def test_active_fusion_artifacts_are_aligned(self):
        config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        if config.get("active_index") != "fusion":
            self.skipTest("Aktif indeks fusion değil")
        reference_names = None
        for variant in config["fusion"]["variants"]:
            directory = ROOT / "index" / "variants" / variant
            required = ["cad_embeddings.npy", "cad_filenames.json", "faiss.index", "meta.json"]
            if not all((directory / name).exists() for name in required):
                self.skipTest(f"Yerel indeks artefaktı eksik: {variant}")
            names = json.loads((directory / "cad_filenames.json").read_text(encoding="utf-8"))
            embeddings = np.load(directory / "cad_embeddings.npy", mmap_mode="r")
            index = faiss.deserialize_index(
                np.fromfile(directory / "faiss.index", dtype=np.uint8)
            )
            self.assertEqual(len(names), embeddings.shape[0], variant)
            self.assertEqual(len(names), index.ntotal, variant)
            self.assertEqual(embeddings.shape[1], index.d, variant)
            self.assertEqual(len(names), len(set(names)), f"Tekrarlanan CAD adı: {variant}")
            self.assertEqual(512, embeddings.shape[1], variant)
            if reference_names is None:
                reference_names = names
            else:
                self.assertEqual(reference_names, names, f"Füzyon sırası farklı: {variant}")


if __name__ == "__main__":
    unittest.main()
