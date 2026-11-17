"""Offline smoke and cell-compilation tests for the Chronos-2 Colab notebooks."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_FILES = {
    "gerador": ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv",
    "gearbox": ROOT / "data" / "Features_gearbox_score_todos_RUL-porcentagem-data.csv",
}
EXPECTED = {
    "gerador": (1910, 574, 4),
    "gearbox": (2946, 822, 3),
}

NOTEBOOK_SPECS = {
    "elghoussani": {
        "path": ROOT / "RUL_CPFL_Chronos2_ElGhoussani_colab.ipynb",
        "config_cell_id": "elghoussani-config",
    },
    "abdouni": {
        "path": ROOT / "RUL_CPFL_Chronos2_Abdouni_colab.ipynb",
        "config_cell_id": "abdouni-config",
    },
}


def _strip_magics(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
    )


def run_chronos_notebook_smoke(
    name: str,
    dataset: str,
    out_dir: Path,
    config_overrides: dict | None = None,
) -> dict:
    spec = NOTEBOOK_SPECS[name]
    cells = json.loads(spec["path"].read_text(encoding="utf-8"))["cells"]
    namespace: dict = {"__name__": f"notebook_{name}"}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        code = _strip_magics("".join(cell["source"]))
        exec(compile(code, f"{spec['path'].name}:{cell.get('id')}", "exec"), namespace)
        if cell.get("id") == spec["config_cell_id"]:
            cfg = namespace["CONFIG"]
            cfg["dataset"] = dataset
            cfg["mode"] = "smoke"
            cfg["input_csv"] = str(CSV_FILES[dataset])
            cfg["output_dir"] = str(out_dir)
            cfg["cache_dir"] = str(out_dir / "cache")
            cfg["install_dependencies"] = False
            cfg["mount_drive"] = False
            cfg["download_zip"] = False
            if config_overrides:
                cfg.update(config_overrides)
    return namespace


class TestChronosNotebooks(unittest.TestCase):
    def test_compile_all_notebook_code_cells(self) -> None:
        for name, spec in NOTEBOOK_SPECS.items():
            self.assertTrue(spec["path"].is_file(), f"Missing {spec['path']}")
            cells = json.loads(spec["path"].read_text(encoding="utf-8"))["cells"]
            for cell in cells:
                if cell["cell_type"] != "code":
                    continue
                code = _strip_magics("".join(cell["source"]))
                compile(code, f"{spec['path'].name}:{cell.get('id')}", "exec")

    def test_default_config_does_not_point_to_drive_when_unmounted(self) -> None:
        for name, spec in NOTEBOOK_SPECS.items():
            cells = json.loads(spec["path"].read_text(encoding="utf-8"))["cells"]
            namespace: dict = {"__name__": f"notebook_{name}_cfg"}
            for cell in cells:
                if cell["cell_type"] != "code":
                    continue
                code = _strip_magics("".join(cell["source"]))
                exec(compile(code, f"{spec['path'].name}:{cell.get('id')}", "exec"), namespace)
                if cell.get("id") == spec["config_cell_id"]:
                    break
            cfg = namespace["CONFIG"]
            self.assertFalse(cfg["mount_drive"])
            self.assertFalse(str(cfg["output_dir"]).startswith("/content/drive"))
            self.assertIsNone(cfg["cache_dir"])

    def test_unmounted_drive_path_fails_without_creating_dir(self) -> None:
        dataset = "gerador"
        if not CSV_FILES[dataset].is_file():
            self.skipTest(f"{dataset} CSV not available")
        for name in NOTEBOOK_SPECS:
            with tempfile.TemporaryDirectory() as tmp:
                fake_drive_root = Path(tmp) / "fake_drive"
                fake_out = fake_drive_root / "MyDrive" / "cer_out"
                with self.assertRaises(RuntimeError) as ctx:
                    run_chronos_notebook_smoke(
                        name,
                        dataset,
                        fake_out,
                        config_overrides={
                            "drive_mount_root": str(fake_drive_root),
                            "output_dir": str(fake_out),
                            "cache_dir": str(fake_out / "cache"),
                        },
                    )
                msg = str(ctx.exception).lower()
                self.assertTrue("drive" in msg or "mount" in msg)
                self.assertFalse(fake_drive_root.exists())

    def test_unmounted_drive_relative_and_symlink_paths_in_notebooks(self) -> None:
        dataset = "gerador"
        if not CSV_FILES[dataset].is_file():
            self.skipTest(f"{dataset} CSV not available")
        for name in NOTEBOOK_SPECS:
            with tempfile.TemporaryDirectory() as tmp:
                cwd_path = Path(tmp)
                fake_drive_root = cwd_path / "drive"
                sym_alias = cwd_path / "drive_link"
                sym_alias.symlink_to(fake_drive_root)
                orig_cwd = Path.cwd()
                try:
                    os.chdir(cwd_path)
                    for bad_out in (
                        "drive/MyDrive/rel_out",
                        str(fake_drive_root / "MyDrive" / "abs_out"),
                        "drive_link/MyDrive/sym_out",
                    ):
                        with self.assertRaises(RuntimeError) as ctx:
                            run_chronos_notebook_smoke(
                                name,
                                dataset,
                                Path(bad_out),
                                config_overrides={
                                    "drive_mount_root": str(fake_drive_root),
                                    "output_dir": bad_out,
                                    "cache_dir": None,
                                },
                            )
                        msg = str(ctx.exception).lower()
                        self.assertTrue("drive" in msg or "mount" in msg)
                        self.assertFalse(fake_drive_root.exists())
                finally:
                    os.chdir(orig_cwd)

    def test_elghoussani_smoke_both_datasets(self) -> None:
        self._assert_smoke_both_datasets("elghoussani")

    def test_abdouni_smoke_both_datasets(self) -> None:
        self._assert_smoke_both_datasets("abdouni")

    def _assert_smoke_both_datasets(self, notebook_name: str) -> None:
        for dataset in ("gerador", "gearbox"):
            if not CSV_FILES[dataset].is_file():
                self.skipTest(f"{dataset} CSV not available")
            clean_keys, endpoints, outer_count = EXPECTED[dataset]
            with tempfile.TemporaryDirectory() as tmp:
                ns = run_chronos_notebook_smoke(notebook_name, dataset, Path(tmp))
                res = ns["RESULT"]
                self.assertEqual(res["status"], "smoke")
                self.assertEqual(res["dataset"], dataset)
                self.assertEqual(res["clean_keys"], clean_keys)
                self.assertEqual(res["endpoint_count"], endpoints)
                self.assertEqual(len(res["outer_folds"]), outer_count)
                self.assertTrue(Path(res["zip_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
