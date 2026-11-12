"""Smoke sem modelos dos dois notebooks Colab, nos dois datasets.

Cada notebook é executado célula a célula, na ordem, com ``CONFIG`` ajustado
logo após a célula de configuração e antes de qualquer efeito (entrada,
saída local, pré-processamento). Nenhum head é ajustado: ``Ridge.fit``,
``SVR.fit`` e ``ExtraTreesRegressor.fit`` são protegidos por um guarda que
levanta erro se forem chamados. Não há Drive, upload, instalação, download,
forecast nem import de torch/Timestral.

Os CSVs reais ficam em ``data/`` e são somente lidos. Pastas temporárias
recebem cópias dos CSVs quando o teste precisa de um diretório com os dois
arquivos. Os testes reais são pulados se o CSV não existir.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTEBOOKS = {
    "baselines": ROOT / "RUL_CPFL_baselines_colab.ipynb",
    "timestral": ROOT / "RUL_CPFL_timestral_colab.ipynb",
}
CONFIG_CELL_ID = {"baselines": "baseline-config", "timestral": "timestral-config"}
CSV_FILES = {
    "gerador": ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv",
    "gearbox": ROOT / "data" / "Features_gearbox_score_todos_RUL-porcentagem-data.csv",
}
EXPECTED = {
    # dataset: (raw rows, clean keys, endpoints, folds, baseline runs, timestral runs,
    #           unit_kind, group_count)
    "gerador": (2710, 1910, 574, 4, 16, 8, "equipment", 4),
    "gearbox": (2946, 2946, 822, 3, 12, 6, "event", 3),
}


def _refuse_fit(*args, **kwargs):
    raise AssertionError("ajuste de modelo chamado em smoke sem treino")


def run_notebook(name, dataset, output_dir, *, preprocessing=None,
                 input_override=None):
    """Executa o notebook inteiro (modo dados, sem treino) e devolve o namespace."""
    cells = json.loads(NOTEBOOKS[name].read_text(encoding="utf-8"))["cells"]
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR

    namespace = {"__name__": f"notebook_{name}"}
    # Apenas exibição: o IPython 'display' não existe fora do Jupyter/Colab.
    namespace["display"] = lambda value: None
    patches = [
        mock.patch.object(Ridge, "fit", _refuse_fit),
        mock.patch.object(SVR, "fit", _refuse_fit),
        mock.patch.object(ExtraTreesRegressor, "fit", _refuse_fit),
    ]
    for patch in patches:
        patch.start()
    try:
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(compile(source, f"{NOTEBOOKS[name].name}:{cell['id']}", "exec"),
                 namespace)
            if cell["id"] == CONFIG_CELL_ID[name]:
                config = namespace["CONFIG"]
                config["run_training"] = False
                config["dataset"] = dataset
                config["input"]["mode"] = "files"
                config["input"]["csv_paths"] = [str(CSV_FILES[dataset])]
                config["output"].update({
                    "mode": "local",
                    "local_dir": str(output_dir),
                    "local_only_acknowledged": True,
                    "download_zip": False,
                    "install_dependencies": False,
                })
                if name == "timestral":
                    config["model_specific"].update({
                        "run_forecast": False,
                        "forecast_smoke": False,
                        "install_dependencies": False,
                    })
                if preprocessing is not None:
                    config["preprocessing"] = dict(preprocessing)
                if input_override is not None:
                    config["input"].update(input_override)
    finally:
        for patch in patches:
            patch.stop()
    return namespace


def _csv_available(dataset):
    return CSV_FILES[dataset].is_file()


class NotebookSmokeBase(unittest.TestCase):
    """Cache de execuções: cada combinação notebook × dataset roda uma vez."""

    _cache = {}
    _tmp = None

    @classmethod
    def setUpClass(cls):
        if NotebookSmokeBase._tmp is None:
            NotebookSmokeBase._tmp = tempfile.mkdtemp(prefix="rul-notebook-tests-")
        cls._tmp = NotebookSmokeBase._tmp

    @classmethod
    def tearDownClass(cls):
        pass

    def run_case(self, name, dataset, preprocessing=None, input_override=None):
        """Execuções de smoke são cacheadas por processo e reutilizadas entre classes."""
        key = (name, dataset, json.dumps(preprocessing, sort_keys=True),
               json.dumps(input_override, sort_keys=True))
        if key not in NotebookSmokeBase._cache:
            output = Path(self._tmp) / f"{name}-{dataset}-{len(NotebookSmokeBase._cache)}"
            NotebookSmokeBase._cache[key] = run_notebook(
                name, dataset, output, preprocessing=preprocessing,
                input_override=input_override)
        return NotebookSmokeBase._cache[key]


class BaselinesNotebookSmokeTests(NotebookSmokeBase):
    def _assert_smoke(self, dataset):
        if not _csv_available(dataset):
            self.skipTest(f"CSV real do dataset {dataset} não disponível em data/")
        raw_rows, clean_keys, endpoints, folds, runs, _, unit_kind, groups = EXPECTED[dataset]
        ns = self.run_case("baselines", dataset)
        self.assertEqual(ns["DATASET_ID"], dataset)
        self.assertEqual(ns["LABEL_ORDER"], ["porcentagem-data"])
        self.assertEqual(len(ns["SPLITS"]), folds)
        self.assertEqual(len(ns["SPLITS"]), groups)
        self.assertEqual(ns["EXPECTED_RAW_ROWS"], raw_rows)
        self.assertEqual(ns["AUDITS"]["porcentagem-data"]["raw_rows"], raw_rows)
        self.assertEqual(ns["AUDITS"]["porcentagem-data"]["clean_keys"], clean_keys)
        self.assertEqual(ns["AUDITS"]["porcentagem-data"]["target_conflict_keys"], 0)
        self.assertEqual(len(ns["ENDPOINTS"]), endpoints)
        self.assertEqual(ns["expected_run_count"](), runs)
        self.assertEqual(ns["FIT_COUNTER"], {})
        self.assertEqual(ns["PROTOCOL"]["unit_kind"], unit_kind)
        self.assertEqual(ns["PROTOCOL"]["labels"]["porcentagem-data"]["max_allowed"], 1.0)
        self.assertEqual(ns["PROTOCOL"]["labels"]["porcentagem-data"]["metric_factor"], 100.0)
        self.assertTrue(Path(ns["RESULTS_EXPORT"]["zip_path"]).is_file())
        manifest = ns["RESULTS_EXPORT"]["manifest"]
        self.assertEqual(manifest["status"], "smoke_without_training")
        self.assertEqual(manifest["dataset"]["dataset_id"], dataset)
        self.assertEqual(manifest["dataset"]["unit_kind"], unit_kind)
        self.assertEqual(manifest["dataset"]["group_count"], groups)
        self.assertEqual(manifest["training"]["expected_runs"], runs)
        self.assertEqual(manifest["training"]["actual_fit_counts"], {})
        self.assertEqual(manifest["endpoints"]["count"], endpoints)
        self.assertEqual(manifest["data"]["input_files"][0]["filename"],
                         CSV_FILES[dataset].name)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("timestral", sys.modules)

    def test_gerador_smoke(self):
        self._assert_smoke("gerador")

    def test_gearbox_smoke(self):
        self._assert_smoke("gearbox")


class TimestralNotebookSmokeTests(NotebookSmokeBase):
    def _assert_smoke(self, dataset):
        if not _csv_available(dataset):
            self.skipTest(f"CSV real do dataset {dataset} não disponível em data/")
        raw_rows, clean_keys, endpoints, folds, _, runs, unit_kind, groups = EXPECTED[dataset]
        ns = self.run_case("timestral", dataset)
        self.assertEqual(ns["DATASET_ID"], dataset)
        self.assertEqual(len(ns["ENDPOINTS"]), endpoints)
        self.assertEqual(ns["AUDITS"]["porcentagem-data"]["clean_keys"], clean_keys)
        self.assertEqual(len(ns["SPLITS"]), folds)
        self.assertEqual(ns["expected_run_count"](), runs)
        self.assertEqual(ns["FIT_COUNTER"], {})
        manifest = ns["RESULTS_EXPORT"]["manifest"]
        self.assertEqual(manifest["status"], "data_smoke_without_training_or_forecast")
        self.assertEqual(manifest["dataset"]["unit_kind"], unit_kind)
        self.assertEqual(manifest["dataset"]["group_count"], groups)
        self.assertEqual(manifest["training"]["expected_runs"], runs)
        self.assertEqual(manifest["training"]["actual_fit_counts"], {})
        self.assertFalse(manifest["forecast"]["model_specific"]["run_forecast"])
        self.assertFalse(manifest["forecast"]["model_specific"]["forecast_smoke"])
        self.assertEqual(manifest["forecast"]["actual_counts"], {})
        self.assertTrue(Path(ns["RESULTS_EXPORT"]["zip_path"]).is_file())
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("timestral", sys.modules)

    def test_gerador_smoke(self):
        self._assert_smoke("gerador")

    def test_gearbox_smoke(self):
        self._assert_smoke("gearbox")


class DatasetSelectionTests(NotebookSmokeBase):
    def test_folder_with_both_csvs_selects_matching_dataset(self):
        if not (_csv_available("gerador") and _csv_available("gearbox")):
            self.skipTest("CSVs reais dos dois datasets não disponíveis em data/")
        folder = Path(self._tmp) / "both-datasets"
        folder.mkdir(parents=True, exist_ok=True)
        for dataset in ("gerador", "gearbox"):
            shutil.copy(CSV_FILES[dataset], folder / CSV_FILES[dataset].name)
        for dataset in ("gerador", "gearbox"):
            with self.subTest(dataset=dataset):
                ns = run_notebook(
                    "baselines", dataset, Path(self._tmp) / f"folder-{dataset}",
                    input_override={"mode": "drive_dir", "drive_dir": str(folder),
                                    "csv_paths": [], "zip_paths": []})
                selected = ns["INPUT_FILES"]["porcentagem-data"]
                self.assertEqual(selected.name, CSV_FILES[dataset].name)
                self.assertEqual(len(ns["RAW"]["porcentagem-data"]),
                                 EXPECTED[dataset][0])

    def test_explicit_csv_of_other_dataset_fails_clearly(self):
        if not (_csv_available("gerador") and _csv_available("gearbox")):
            self.skipTest("CSVs reais dos dois datasets não disponíveis em data/")
        with self.assertRaises(ValueError) as context:
            run_notebook("baselines", "gearbox", Path(self._tmp) / "wrong-file",
                         input_override={"csv_paths": [str(CSV_FILES["gerador"])]})
        self.assertIn("não é o arquivo do dataset selecionado", str(context.exception))

    def test_invalid_preprocessing_config_is_rejected_early(self):
        for bad in ({"method": "bogus", "add_missing_indicators": True},
                    {"method": "standard", "add_missing_indicators": "False"},
                    {"method": "standard"}):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                run_notebook("baselines", "gerador", Path(self._tmp) / "bad-preproc",
                             preprocessing=bad)


class AggregationTagTests(NotebookSmokeBase):
    """Regressão: a tag de agregação macro reflete unidade e número real de grupos.

    Usa linhas sintéticas fixas (não são escores científicos) e as execuções
    de smoke em cache dos dois notebooks. Não ajusta modelos.
    """

    @staticmethod
    def _synthetic_prediction_frame(ns):
        # Dois folds, valores fixos. Só testam a semântica de agregação.
        labels = ns["LABEL_ORDER"]
        unit = ns["PROTOCOL"]["labels"][labels[0]]["unit"]
        rows = []
        for index, group in enumerate(ns["EQUIPMENT_ORDER"][:2]):
            members = ns["PROTOCOL"]["equipment_groups"][group]
            falha = int(members[0])
            rows.append({
                "label": labels[0], "family": "ridge", "unit": unit,
                "fold_id": f"fold_{index}", "test_equipment": group,
                "endpoint_id": f"synthetic-{index}", "Falha": falha,
                "equipment": ns["ID_TO_EQUIPMENT"][falha], "phase": "saudavel",
                "y_metric": 10.0 + index, "pred_metric": 12.0 + 2 * index,
            })
        return pd.DataFrame(rows)

    def _check(self, name, dataset, expected_tag):
        if not _csv_available(dataset):
            self.skipTest(f"CSV real do dataset {dataset} não disponível em data/")
        ns = self.run_case(name, dataset)
        frame = self._synthetic_prediction_frame(ns)
        rows = ns["fold_metric_rows"](frame)
        macro = ns["aggregate_metric_rows"](rows)
        equipment = macro[macro["scope"] == "equipment_macro"]
        self.assertFalse(equipment.empty)
        self.assertTrue((equipment["aggregation"] == expected_tag).all())
        if dataset == "gearbox":
            tags = " ".join(macro["aggregation"].astype(str))
            self.assertNotIn("4_equipamentos", tags)
        # Overall (pooled entre folds) mantém a tag própria, sem trocar.
        overall = macro[macro["scope"] == "overall"]
        self.assertTrue((overall["aggregation"] == "media_simples_entre_folds_por_familia").all())
        # Numérico idêntico ao módulo; só a tag muda.
        legacy = ns["_cer_rul"].aggregate_metric_rows(rows)
        numeric = [c for c in macro.columns if c != "aggregation"]
        pd.testing.assert_frame_equal(macro[numeric], legacy[numeric])
        # CSV exportado não contém a tag errada.
        target = Path(self._tmp) / f"macro-{name}-{dataset}.csv"
        ns["write_csv"](macro, target)
        text = target.read_text(encoding="utf-8")
        self.assertIn(expected_tag, text)
        if dataset == "gearbox":
            self.assertNotIn("4_equipamentos", text)
        return macro

    def test_baselines_gerador_tag(self):
        self._check("baselines", "gerador",
                    "media_igual_entre_4_equipamentos_por_familia")

    def test_baselines_gearbox_tag_is_eventos(self):
        self._check("baselines", "gearbox",
                    "media_igual_entre_3_eventos_por_familia")

    def test_timestral_gerador_tag(self):
        self._check("timestral", "gerador",
                    "media_igual_entre_4_equipamentos_por_familia")

    def test_timestral_gearbox_tag_is_eventos(self):
        self._check("timestral", "gearbox",
                    "media_igual_entre_3_eventos_por_familia")


class IdentityTests(NotebookSmokeBase):
    def test_dataset_and_preprocessing_change_identities(self):
        if not (_csv_available("gerador") and _csv_available("gearbox")):
            self.skipTest("CSVs reais dos dois datasets não disponíveis em data/")
        base = self.run_case("baselines", "gerador")
        robust = self.run_case("baselines", "gerador",
                               preprocessing={"method": "robust",
                                              "add_missing_indicators": True})
        no_indicators = self.run_case("baselines", "gerador",
                                      preprocessing={"method": "standard",
                                                     "add_missing_indicators": False})
        gearbox = self.run_case("baselines", "gearbox")
        run_ids = {base["RUN_ID"], robust["RUN_ID"], no_indicators["RUN_ID"],
                   gearbox["RUN_ID"]}
        self.assertEqual(len(run_ids), 4)

        fold = base["SPLITS"].to_dict(orient="records")[0]
        signatures = {
            name: ns["search_signature"]("porcentagem-data", fold, "ridge")[0]
            for name, ns in (("base", base), ("robust", robust),
                             ("no_indicators", no_indicators))
        }
        self.assertEqual(len(set(signatures.values())), 3)

        self.assertNotEqual(base["DATA_IDENTITY_HASH"], gearbox["DATA_IDENTITY_HASH"])
        self.assertNotEqual(base["PROTOCOL_HASH"], gearbox["PROTOCOL_HASH"])

    def test_timestral_forecast_cache_depends_on_dataset(self):
        if not (_csv_available("gerador") and _csv_available("gearbox")):
            self.skipTest("CSVs reais dos dois datasets não disponíveis em data/")
        keys = {
            dataset: self.run_case("timestral", dataset)["timestral_cache_key"](12, 3.0)
            for dataset in ("gerador", "gearbox")
        }
        self.assertNotEqual(keys["gerador"], keys["gearbox"])


def tearDownModule():
    if NotebookSmokeBase._tmp is not None:
        shutil.rmtree(NotebookSmokeBase._tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
