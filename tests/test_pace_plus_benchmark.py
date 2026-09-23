from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "verl" / "verl" / "trainer"

verl_package = ModuleType("verl")
verl_package.__path__ = [str(ROOT / "verl" / "verl")]
sys.modules.setdefault("verl", verl_package)
trainer_package = ModuleType("verl.trainer")
trainer_package.__path__ = [str(TRAINER)]
sys.modules.setdefault("verl.trainer", trainer_package)

import verl.trainer.pace_plus_benchmark as benchmark_module
from verl.trainer.pace_plus_benchmark import (
    BenchmarkDataset,
    _append_prediction_records,
    _compact_predictions,
    _generate_reasoning_batch,
    _load_teacher_experience_context,
    _plan_dataset,
    load_benchmark_datasets,
)
from verl.trainer.pace_plus_config import load_config, project_path
from verl.trainer.pace_plus_metrics import classification_metrics
from verl.trainer.pace_plus_schema import SarcasmSample


VALID_REASONING = {
    "visual_evidence": "A person smiles while holding a broken umbrella in heavy rain.",
    "textual_evidence": "The post calls this perfect weather.",
    "explanation": "Surface meaning: Praise; Intended meaning: Criticism; Judgment rationale: The evidence is incongruous.",
    "label": "sarcastic",
}


class BenchmarkDatasetTests(unittest.TestCase):
    def test_all_configured_test_sets_load(self):
        config = load_config(ROOT / "configs" / "pace_plus_msd.yaml")
        for spec in config["benchmark_evaluation"]["datasets"].values():
            paths = [spec["test_path"]] if "test_path" in spec else spec["test_paths"].values()
            if any(not project_path(config, path).is_file() for path in paths):
                self.skipTest("install the optional benchmark datasets to run this integration test")
        datasets = load_benchmark_datasets(config, limit=2)
        self.assertEqual(
            [dataset.name for dataset in datasets],
            ["mmsd2", "sarcnet", "docmsu", "redeval"],
        )
        self.assertTrue(all(len(dataset.samples) == 2 for dataset in datasets))
        self.assertTrue(
            all(Path(sample.image_path).is_file() for dataset in datasets for sample in dataset.samples)
        )


class BenchmarkResumeTests(unittest.TestCase):
    def _sample(self, directory: str) -> SarcasmSample:
        image = Path(directory) / "image.jpg"
        image.write_bytes(b"image")
        return SarcasmSample(
            sample_id="sample-1",
            image_path=str(image),
            text="Perfect weather.",
            label="sarcastic",
            language="en",
        )

    def test_completed_journal_batch_is_reused_and_compacted(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = self._sample(directory)
            dataset = BenchmarkDataset("mmsd2", [sample], ["test.json"])
            output_root = Path(directory) / "outputs"
            state = _plan_dataset(
                dataset=dataset,
                output_root=output_root,
                role="teacher",
                base_fingerprint="model-and-prompt",
                retry_invalid=False,
            )
            self.assertEqual(len(state.pending), 1)
            fingerprint = state.fingerprints[sample.sample_id]
            record = {
                "sample_id": sample.sample_id,
                "analysis": VALID_REASONING,
                "json_valid": True,
                "error": None,
                "fingerprint": fingerprint,
            }
            _append_prediction_records(state.journal_path, [record])

            resumed = _plan_dataset(
                dataset=dataset,
                output_root=output_root,
                role="teacher",
                base_fingerprint="model-and-prompt",
                retry_invalid=False,
            )
            self.assertEqual(resumed.pending, [])
            _compact_predictions(resumed)
            self.assertFalse(resumed.journal_path.exists())
            self.assertEqual(
                json.loads(resumed.output_path.read_text(encoding="utf-8").strip())["sample_id"],
                sample.sample_id,
            )

    def test_capacity_upgrade_migrates_only_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = self._sample(directory)
            dataset = BenchmarkDataset("mmsd2", [sample], ["test.json"])
            output_root = Path(directory) / "outputs"
            old_state = _plan_dataset(
                dataset=dataset,
                output_root=output_root,
                role="teacher",
                base_fingerprint="old-capacity",
                retry_invalid=True,
            )
            old_fingerprint = old_state.fingerprints[sample.sample_id]
            _append_prediction_records(
                old_state.journal_path,
                [
                    {
                        "sample_id": sample.sample_id,
                        "analysis": VALID_REASONING,
                        "json_valid": True,
                        "error": None,
                        "fingerprint": old_fingerprint,
                    }
                ],
            )

            migrated = _plan_dataset(
                dataset=dataset,
                output_root=output_root,
                role="teacher",
                base_fingerprint="new-capacity",
                retry_invalid=True,
                compatible_base_fingerprints=["old-capacity"],
            )
            self.assertEqual(migrated.pending, [])
            record = migrated.records[sample.sample_id]
            self.assertEqual(
                record["fingerprint"],
                migrated.fingerprints[sample.sample_id],
            )
            self.assertEqual(record["cache_migrated_from"], old_fingerprint)

    def test_truncated_journal_tail_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = self._sample(directory)
            dataset = BenchmarkDataset("mmsd2", [sample], ["test.json"])
            state = _plan_dataset(
                dataset=dataset,
                output_root=Path(directory) / "outputs",
                role="teacher",
                base_fingerprint="model-and-prompt",
                retry_invalid=False,
            )
            _append_prediction_records(
                state.journal_path,
                [
                    {
                        "sample_id": sample.sample_id,
                        "analysis": VALID_REASONING,
                        "json_valid": True,
                        "error": None,
                        "fingerprint": state.fingerprints[sample.sample_id],
                    }
                ],
            )
            with state.journal_path.open("a", encoding="utf-8") as handle:
                handle.write('{"sample_id": "interrupted"')

            resumed = _plan_dataset(
                dataset=dataset,
                output_root=Path(directory) / "outputs",
                role="teacher",
                base_fingerprint="model-and-prompt",
                retry_invalid=False,
            )
            self.assertEqual(resumed.pending, [])
            lines = resumed.journal_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["sample_id"], sample.sample_id)

    def test_invalid_cache_is_optional_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = self._sample(directory)
            dataset = BenchmarkDataset("redeval", [sample], ["test.json"])
            output_root = Path(directory) / "outputs"
            state = _plan_dataset(
                dataset=dataset,
                output_root=output_root,
                role="student",
                base_fingerprint="model-and-prompt",
                retry_invalid=False,
            )
            _append_prediction_records(
                state.journal_path,
                [
                    {
                        "sample_id": sample.sample_id,
                        "analysis": None,
                        "json_valid": False,
                        "error": "invalid",
                        "fingerprint": state.fingerprints[sample.sample_id],
                    }
                ],
            )
            cached = _plan_dataset(
                dataset=dataset,
                output_root=output_root,
                role="student",
                base_fingerprint="model-and-prompt",
                retry_invalid=False,
            )
            retry = _plan_dataset(
                dataset=dataset,
                output_root=output_root,
                role="student",
                base_fingerprint="model-and-prompt",
                retry_invalid=True,
            )
            self.assertEqual(cached.pending, [])
            self.assertEqual(len(retry.pending), 1)


class BenchmarkBatchTests(unittest.TestCase):
    class Generator:
        def __init__(self):
            self.outputs = [
                ["bad JSON", json.dumps(VALID_REASONING)],
                [json.dumps(VALID_REASONING)],
            ]
            self.batch_sizes: list[int] = []
            self.requests = []

        def generate_batch(self, requests):
            self.batch_sizes.append(len(requests))
            self.requests.append(requests)
            return self.outputs.pop(0)

    def test_batch_retry_preserves_all_four_reasoning_fields(self):
        samples = [
            SarcasmSample(
                sample_id=f"sample-{index}",
                image_path=f"test-assets/image-{index}.jpg",
                text="A post.",
                label="sarcastic",
            )
            for index in range(2)
        ]
        generator = self.Generator()
        with tempfile.TemporaryDirectory() as directory:
            records = _generate_reasoning_batch(
                generator=generator,
                batch=[
                    (samples[0], "fingerprint-0"),
                    (samples[1], "fingerprint-1"),
                ],
                role="teacher",
                dataset="mmsd2",
                model_path=Path("models/test-model"),
                failure_log_path=Path(directory) / "failures.jsonl",
                max_attempts=3,
            )
        self.assertEqual(generator.batch_sizes, [2, 1])
        self.assertIn("Never copy a double quote", json.dumps(generator.requests[1]))
        self.assertEqual([record["attempts"] for record in records], [2, 1])
        self.assertTrue(all(record["json_valid"] for record in records))
        self.assertTrue(
            all(
                list(record["analysis"]) == [
                    "visual_evidence",
                    "textual_evidence",
                    "explanation",
                    "label",
                ]
                for record in records
            )
        )


    def test_teacher_experience_is_injected_without_exposing_gold(self):
        sample = SarcasmSample(
            sample_id="sample-0",
            image_path="test-assets/image-0.jpg",
            text="A post.",
            label="sarcastic",
        )
        generator = self.Generator()
        generator.outputs = [[json.dumps(VALID_REASONING)]]
        with tempfile.TemporaryDirectory() as directory:
            records = _generate_reasoning_batch(
                generator=generator,
                batch=[(sample, "fingerprint-0")],
                role="teacher",
                dataset="redeval",
                model_path=Path("models/test-model"),
                failure_log_path=Path(directory) / "failures.jsonl",
                max_attempts=3,
                experience="Use cross-modal incongruity as evidence.",
            )
        serialized_prompt = json.dumps(generator.requests[0])
        self.assertIn("Use cross-modal incongruity as evidence.", serialized_prompt)
        self.assertNotIn("gold_label", serialized_prompt)
        self.assertNotIn("reference_reasoning", serialized_prompt)
        self.assertIsNotNone(records[0]["teacher_experience_sha256"])


class TeacherExperienceTests(unittest.TestCase):
    def test_active_experience_count_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            path.write_text(
                json.dumps(
                    {
                        "experiences": [
                            {"id": "active", "text": "Use incongruity.", "active": True},
                            {"id": "inactive", "text": "Ignore this.", "active": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            experience, metadata = _load_teacher_experience_context(
                {}, path, expected_active=1
            )
            self.assertEqual(experience, "- Use incongruity.")
            self.assertEqual(metadata["active_count"], 1)
            with self.assertRaisesRegex(Exception, "expected 2"):
                _load_teacher_experience_context({}, path, expected_active=2)



class BenchmarkEndToEndTests(unittest.TestCase):
    class Generator:
        loads = 0

        def __init__(self, *args, **kwargs):
            del args, kwargs
            type(self).loads += 1

        def generate_batch(self, requests):
            return [json.dumps(VALID_REASONING) for _ in requests]

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def test_same_command_resumes_without_reloading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "configs" / "pace_plus_msd.yaml")
            teacher_model = Path(
                os.environ.get(
                    "PACE_PLUS_TEACHER_MODEL",
                    config["models"]["teacher"]["path"],
                )
            )
            if not teacher_model.is_absolute():
                teacher_model = ROOT / teacher_model
            config["models"]["teacher"]["path"] = str(teacher_model)
            if not (teacher_model / "config.json").is_file():
                self.skipTest(
                    "provide PACE_PLUS_TEACHER_MODEL or models/Qwen3.5-9B "
                    "to run checkpoint-backed benchmark tests"
                )
            dataset_path = config["benchmark_evaluation"]["datasets"]["mmsd2"]["test_path"]
            if not project_path(config, dataset_path).is_file():
                self.skipTest("install the optional MMSD2.0 dataset to run this integration test")
            config["benchmark_evaluation"]["output_path"] = str(Path(directory) / "benchmark")
            config["benchmark_evaluation"]["pause_file"] = str(Path(directory) / "benchmark" / ".pause")
            original_device = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
            try:
                self.Generator.loads = 0
                with mock.patch.object(
                    benchmark_module,
                    "VllmMultimodalGenerator",
                    self.Generator,
                ):
                    first = benchmark_module.run_benchmark(
                        config,
                        devices=["1"],
                        roles=["teacher"],
                        datasets=["mmsd2"],
                        limit=2,
                    )
                self.assertEqual(first["status"], "complete")
                self.assertEqual(self.Generator.loads, 1)
                self.assertTrue((Path(directory) / "benchmark" / "summary.json").is_file())
                self.assertTrue((Path(directory) / "benchmark" / "summary.csv").is_file())
                primary_path = Path(directory) / "benchmark" / "primary_metrics.csv"
                self.assertTrue(primary_path.is_file())
                self.assertEqual(
                    primary_path.read_text(encoding="utf-8").splitlines()[0],
                    "role,dataset,dataset_samples,accuracy,precision,recall,macro_f1",
                )

                class FailIfLoaded:
                    def __init__(self, *args, **kwargs):
                        del args, kwargs
                        raise AssertionError("completed predictions must be resumed")

                with mock.patch.object(
                    benchmark_module,
                    "VllmMultimodalGenerator",
                    FailIfLoaded,
                ):
                    resumed = benchmark_module.run_benchmark(
                        config,
                        devices=["1"],
                        roles=["teacher"],
                        datasets=["mmsd2"],
                        limit=2,
                    )
                self.assertEqual(resumed["status"], "complete")
            finally:
                if original_device is None:
                    __import__("os").environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    __import__("os").environ["CUDA_VISIBLE_DEVICES"] = original_device


class RichMetricTests(unittest.TestCase):
    def test_rich_metrics_include_confusion_and_abstention_penalty(self):
        result = classification_metrics(
            ["sarcastic", "non-sarcastic"],
            ["sarcastic", "sarcastic"],
            invalid=1,
            invalid_gold=["non-sarcastic"],
        )
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["prediction_coverage"], 2 / 3)
        self.assertIn("macro_precision", result)
        self.assertIn("weighted_f1", result)
        self.assertIn("matthews_correlation_coefficient_valid", result)
        self.assertEqual(result["confusion_matrix_valid"]["true_positive"], 1)
        self.assertEqual(result["invalid_by_gold"]["non-sarcastic"], 1)


if __name__ == "__main__":
    unittest.main()
