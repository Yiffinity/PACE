from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "verl" / "verl" / "trainer"

verl_package = ModuleType("verl")
verl_package.__path__ = [str(ROOT / "verl" / "verl")]
sys.modules.setdefault("verl", verl_package)
trainer_package = ModuleType("verl.trainer")
trainer_package.__path__ = [str(TRAINER)]
sys.modules.setdefault("verl.trainer", trainer_package)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


schema = load_module("verl.trainer.pace_schema", TRAINER / "pace_schema.py")
prompts = load_module("verl.trainer.pace_prompts", TRAINER / "pace_prompts.py")
reflection = load_module("verl.trainer.pace_reflection", TRAINER / "pace_reflection.py")
pool = load_module("verl.trainer.pace_pool", TRAINER / "pace_pool.py")
weighted_pool = load_module(
    "verl.trainer.pace_weighted_pool",
    TRAINER / "pace_weighted_pool.py",
)
extraction = load_module("verl.trainer.pace_extraction", TRAINER / "pace_extraction.py")
modeling = sys.modules["verl.trainer.pace_modeling"]
preflight = load_module("pace_preflight_test", TRAINER / "pace_preflight.py")
metrics = load_module("pace_metrics_test", TRAINER / "pace_metrics.py")
reward = load_module("pace_reward_test", TRAINER / "pace_reward.py")


VALID_REASONING = {
    "visual_evidence": "A smiling person holds a broken umbrella in heavy rain.",
    "textual_evidence": "The post calls this perfect weather.",
    "explanation": "Surface meaning: Praise; Intended meaning: Criticism; Judgment rationale: The modalities conflict.",
    "label": "sarcastic",
}


class SchemaTests(unittest.TestCase):
    def test_reasoning_has_exact_fields(self):
        parsed = schema.validate_reasoning(VALID_REASONING)
        self.assertEqual(parsed.label, "sarcastic")
        with self.assertRaises(schema.SchemaError):
            schema.validate_reasoning({**VALID_REASONING, "confidence": 0.9})
        with self.assertRaises(schema.SchemaError):
            schema.validate_reasoning({**VALID_REASONING, "label": "ironic"})

    def test_parser_rejects_markdown_and_non_object(self):
        with self.assertRaises(schema.SchemaError):
            schema.parse_reasoning("\x60\x60\x60json\n{}\n\x60\x60\x60")
        with self.assertRaises(schema.SchemaError):
            schema.parse_json_object("[]")

    def test_reference_label_must_match_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.touch()
            sample = {
                "sample_id": "sample-1",
                "image_path": str(image),
                "text": "Perfect weather.",
                "label": "non-sarcastic",
                "reference_reasoning": VALID_REASONING,
            }
            with self.assertRaises(schema.SchemaError):
                schema.validate_sample(sample)

    def test_jsonl_upsert_replaces_instead_of_duplicating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            schema.upsert_jsonl(path, {"sample_id": "a", "value": 1})
            schema.upsert_jsonl(path, {"sample_id": "a", "value": 2})
            records = list(schema.read_jsonl(path))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0][1]["value"], 2)


class JsonRetryTests(unittest.TestCase):
    class _Generator:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = 0
            self.requests = []

        def generate(self, messages, *, image_path):
            del image_path
            self.calls += 1
            self.requests.append(messages)
            return self.outputs.pop(0)

    def test_thinking_wrapper_and_retry(self):
        generator = self._Generator([
            "not json",
            "<think>internal analysis</think>\n" + json.dumps(VALID_REASONING),
        ])
        with tempfile.TemporaryDirectory() as directory:
            parsed, raw, error, attempts = schema.generate_json_with_retries(
                generator,
                [{"role": "user", "content": "x"}],
                image_path=None,
                parser=schema.parse_reasoning,
                max_attempts=3,
                failure_log_path=Path(directory) / "failures.jsonl",
                context={"stage": "test", "sample_id": "one"},
            )
        self.assertEqual(parsed.label, "sarcastic")
        self.assertIsNone(error)
        self.assertEqual(attempts, 2)
        self.assertEqual(
            generator.requests[1][-2],
            {
                "role": "assistant",
                "content": (
                    "I will discard the invalid output and generate a fresh JSON object."
                ),
            },
        )
        self.assertIn(
            "at least ten words below",
            generator.requests[1][-1]["content"],
        )
        self.assertEqual(generator.calls, 2)
        self.assertIn("</think>", raw)

    def test_exhausted_retries_are_logged(self):
        generator = self._Generator(["bad", "still bad", "also bad"])
        with tempfile.TemporaryDirectory() as directory:
            failure_path = Path(directory) / "failures.jsonl"
            parsed, raw, error, attempts = schema.generate_json_with_retries(
                generator,
                [{"role": "user", "content": "x"}],
                image_path=None,
                parser=schema.parse_reasoning,
                max_attempts=3,
                failure_log_path=failure_path,
                context={"stage": "test", "sample_id": "two"},
            )
            entries = list(schema.read_jsonl(failure_path))
        self.assertIsNone(parsed)
        self.assertEqual(raw, "also bad")
        self.assertIsNotNone(error)
        self.assertEqual(attempts, 3)
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0][1]["attempts"]), 3)

    def test_cuda_oom_is_fatal_without_retry(self):
        class OOMGenerator:
            calls = 0

            def generate(self, messages, *, image_path):
                del messages, image_path
                self.calls += 1
                raise RuntimeError("CUDA out of memory")

        generator = OOMGenerator()
        with tempfile.TemporaryDirectory() as directory:
            failure_path = Path(directory) / "failures.jsonl"
            with self.assertRaisesRegex(RuntimeError, "fatal accelerator failure"):
                schema.generate_json_with_retries(
                    generator,
                    [{"role": "user", "content": "x"}],
                    image_path=None,
                    parser=schema.parse_reasoning,
                    max_attempts=3,
                    failure_log_path=failure_path,
                    context={"stage": "test", "sample_id": "oom"},
                )
            entries = list(schema.read_jsonl(failure_path))
        self.assertEqual(generator.calls, 1)
        self.assertEqual(len(entries), 1)
        self.assertIs(entries[0][1]["fatal"], True)




class ReasoningBatchTests(unittest.TestCase):
    class _Generator:
        def __init__(self):
            valid = "<think>analysis</think>\n" + json.dumps(VALID_REASONING)
            self.responses = [["not json", valid], [valid]]
            self.batch_sizes = []
            self.requests = []

        def generate_batch(self, requests):
            self.batch_sizes.append(len(requests))
            self.requests.append(requests)
            return self.responses.pop(0)

    def test_batch_retry_only_regenerates_invalid_items(self):
        samples = [
            schema.SarcasmSample(
                sample_id=f"sample-{index}",
                image_path=f"test-assets/image-{index}.jpg",
                text="A post.",
                label="sarcastic",
            )
            for index in range(2)
        ]
        generator = self._Generator()
        with tempfile.TemporaryDirectory() as directory:
            failure_path = Path(directory) / "failures.jsonl"
            records = extraction._generate_reasoning_batch(
                generator=generator,
                batch=[(samples[0], "fingerprint-0"), (samples[1], "fingerprint-1")],
                role="student",
                model_path="models/test-model",
                failure_log_path=failure_path,
                max_attempts=3,
            )
            self.assertFalse(failure_path.exists())

        self.assertEqual(generator.batch_sizes, [2, 1])
        self.assertEqual([record["attempts"] for record in records], [2, 1])
        self.assertTrue(all(record["json_valid"] for record in records))
        self.assertIn("previous response", json.dumps(generator.requests[1]))

    def test_request_value_error_does_not_poison_batch(self):
        class Generator:
            def __init__(self):
                self.batch_sizes = []

            def generate_batch(self, requests):
                self.batch_sizes.append(len(requests))
                if any("oversized" in image_path for _, image_path in requests):
                    raise ValueError("decoder prompt is longer than the maximum model length")
                return [
                    "<think>analysis</think>\n" + json.dumps(VALID_REASONING)
                    for _ in requests
                ]

        samples = [
            schema.SarcasmSample(
                sample_id=f"sample-{index}",
                image_path="test-assets/oversized.jpg" if index == 1 else f"test-assets/image-{index}.jpg",
                text="A post.",
                label="sarcastic",
            )
            for index in range(3)
        ]
        generator = Generator()
        with tempfile.TemporaryDirectory() as directory:
            failure_path = Path(directory) / "failures.jsonl"
            records = extraction._generate_reasoning_batch(
                generator=generator,
                batch=[(sample, f"fingerprint-{index}") for index, sample in enumerate(samples)],
                role="student",
                model_path="models/test-model",
                failure_log_path=failure_path,
                max_attempts=3,
            )
            failures = list(schema.read_jsonl(failure_path))

        self.assertEqual([record["json_valid"] for record in records], [True, False, True])
        self.assertEqual(records[1]["attempts"], 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(generator.batch_sizes[0], 3)
        self.assertIn(1, generator.batch_sizes)

    def test_reasoning_journal_overrides_and_compacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "reasonings.jsonl"
            extraction._atomic_jsonl(
                output_path,
                [{"sample_id": "sample-0", "value": "old"}],
            )
            journal_path = extraction._reasoning_journal_path(output_path)
            extraction._append_reasoning_records(
                journal_path,
                [
                    {"sample_id": "sample-0", "value": "new"},
                    {"sample_id": "sample-1", "value": "added"},
                ],
            )

            records, loaded_journal = extraction._load_reasoning_records(output_path)
            self.assertEqual(records["sample-0"]["value"], "new")
            self.assertEqual(records["sample-1"]["value"], "added")
            extraction._compact_reasoning_records(output_path, loaded_journal, records)

            self.assertFalse(journal_path.exists())
            compacted = schema.load_records_by_id(output_path)
            self.assertEqual(len(compacted), 2)
            self.assertEqual(compacted["sample-0"]["value"], "new")
class PromptTests(unittest.TestCase):
    def test_blind_prompt_contains_image_but_no_gold(self):
        messages = prompts.blind_messages("A post")
        self.assertEqual(messages[1]["content"][0], {"type": "image"})
        serialized = json.dumps(messages)
        self.assertNotIn("gold_label", serialized)
        self.assertNotIn("reference_reasoning", serialized)

    def test_experience_conditioning_preserves_original(self):
        messages = prompts.blind_messages("A post")
        conditioned = prompts.experience_conditioned_messages(
            messages, "Check cross-modal incongruity."
        )
        self.assertEqual(len(conditioned), len(messages))
        self.assertEqual([message["role"] for message in conditioned], ["system", "user"])
        content = conditioned[0]["content"]
        self.assertIn("Check cross-modal incongruity.", content)
        self.assertIn("complete set of previously learned references", content)
        self.assertIn("primary judgment from the raw image and text", content)
        self.assertIn("conditional checks, not as votes", content)
        self.assertIn("within the image or image text", content)
        self.assertIn("agreement does not imply non-sarcasm", content)
        self.assertIn("cross-modal contradiction is not required", content)
        self.assertIn("mechanism followed by its exclusion boundary", content)
        self.assertIn("follow the observable evidence", content)
        self.assertNotIn("non-sarcastic experience", content)
        self.assertEqual(len(messages), 2)
        self.assertEqual(conditioned[-1]["content"][0], {"type": "image"})

    def test_multistage_consolidation_prompts_preserve_confirmed_rules(self):
        normalization = prompts.CANDIDATE_NORMALIZATION_SYSTEM_PROMPT
        self.assertIn("Do not compare it with the reference set", normalization)
        self.assertIn('gold_label = "sarcastic"', normalization)
        self.assertIn('gold_label = "non-sarcastic"', normalization)
        self.assertIn("Never create an independent rule for predicting non-sarcasm", normalization)
        self.assertIn('"mechanism": "English sarcasm mechanism or null"', normalization)
        self.assertIn('"exclusion_boundary": "English exclusion boundary or null"', normalization)

        comparison = prompts.CONSOLIDATION_SYSTEM_PROMPT
        self.assertIn("Compare mechanisms rather than topics", comparison)
        self.assertIn("allowed only for a sarcastic candidate", comparison)
        self.assertIn("only its exclusion boundary may be strengthened", comparison)
        self.assertIn("control outcome allowed only for a non-sarcastic candidate", comparison)
        self.assertIn("Do not write or rewrite a reference in this call", comparison)
        self.assertIn('"action": "ADD or MODIFY or UPVOTE or DOWNVOTE or SKIP"', comparison)
        self.assertIn("Similarity or duplication is supporting evidence", comparison)
        self.assertNotIn("MATCH", comparison)
        self.assertNotIn("HELPFUL", comparison)
        self.assertNotIn("HARMFUL", comparison)

        merge = prompts.EXPERIENCE_MERGE_SYSTEM_PROMPT
        self.assertIn("do not choose an action or another target", merge)
        self.assertIn("Preserve every valid applicability condition", merge)
        self.assertIn('"mechanism": "merged English sarcasm mechanism"', merge)
        self.assertIn("Use up to 95 words", merge)
        self.assertIn("never omit a distinct valid condition", merge)
        self.assertNotIn("150", merge)
        boundary_merge = prompts.EXCLUSION_BOUNDARY_MERGE_SYSTEM_PROMPT
        self.assertIn("target mechanism", boundary_merge)
        self.assertIn("immutable", boundary_merge)
        self.assertIn("revised_exclusion_boundary", boundary_merge)
        self.assertIn("maximum_revised_boundary_words", boundary_merge)
        boundary_compression = (
            prompts.exclusion_boundary_compression_system_prompt(
                target_maximum_words=58,
                compression_strategy="Merge overlapping counterconditions.",
            )
        )
        self.assertIn(
            "no more than 58 space-delimited English words", boundary_compression
        )
        self.assertIn("immutable target mechanism", boundary_compression)
        self.assertIn("causal_categories", boundary_compression)
        self.assertIn("neither a rejected long draft nor the source manifestations", boundary_compression)
        self.assertNotIn("150", boundary_compression)
        abstraction = prompts.exclusion_boundary_abstraction_system_prompt(
            maximum_categories=3,
            compression_strategy="Group by causal role.",
        )
        self.assertIn("source_target", abstraction)
        self.assertIn("source_candidate", abstraction)
        self.assertIn("at most 3", abstraction)
        self.assertIn("does not supply any rejected merged draft", abstraction)
        compression = prompts.experience_compression_system_prompt(
            target_maximum_words=80,
            compression_strategy="Merge overlapping clauses.",
        )
        self.assertIn("no more than 80 space-delimited English words", compression)
        self.assertIn("source_target", compression)
        self.assertIn("source_candidate", compression)
        self.assertIn("does not receive the rejected merged draft", compression)
        self.assertIn("maximum_mechanism_words", compression)
        self.assertIn("count each field's words separately", compression)
        self.assertIn("combine overlapping conditions", compression)
        self.assertIn("surface variants, not distinct conditions", compression)
        self.assertNotIn("150", compression)

    def test_compressed_merge_accepts_tolerance_but_reports_soft_targets(self):
        mechanism = ("Classify as sarcastic when " + "contrast " * 65).strip() + "."
        boundary = ("Do not apply when " + "literal " * 70).strip() + "."
        raw = json.dumps(
            {"mechanism": mechanism, "exclusion_boundary": boundary}
        )
        accepted = weighted_pool._parse_compressed_merge(
            raw,
            accepted_maximum_words=150,
            target_maximum_words=80,
            maximum_mechanism_words=40,
            maximum_boundary_words=40,
        )
        self.assertEqual(len(accepted["experience"].split()), 143)

        over_limit = json.dumps(
            {
                "mechanism": ("Classify as sarcastic when " + "contrast " * 70).strip() + ".",
                "exclusion_boundary": ("Do not apply when " + "literal " * 75).strip() + ".",
            }
        )
        with self.assertRaisesRegex(
            schema.SchemaError,
            r"maximum is 150.*toward 80 words.*target 40",
        ):
            weighted_pool._parse_compressed_merge(
                over_limit,
                accepted_maximum_words=150,
                target_maximum_words=80,
                maximum_mechanism_words=40,
                maximum_boundary_words=40,
            )

    def test_compressed_boundary_accepts_tolerance_but_reports_soft_target(self):
        mechanism = ("Classify as sarcastic when " + "contrast " * 60).strip() + "."
        accepted_raw = json.dumps(
            {"revised_exclusion_boundary": ("Do not apply when " + "literal " * 70).strip() + "."}
        )
        accepted = weighted_pool._parse_compressed_boundary_merge(
            accepted_raw,
            mechanism=mechanism,
            accepted_maximum_words=150,
            target_total_words=95,
            target_boundary_words=31,
        )
        self.assertEqual(len(accepted["experience"].split()), 138)

        over_limit = json.dumps(
            {"revised_exclusion_boundary": ("Do not apply when " + "literal " * 85).strip() + "."}
        )
        with self.assertRaisesRegex(
            schema.SchemaError,
            r"maximum is 150.*boundary toward 31 words.*95 including",
        ):
            weighted_pool._parse_compressed_boundary_merge(
                over_limit,
                mechanism=mechanism,
                accepted_maximum_words=150,
                target_total_words=95,
                target_boundary_words=31,
            )

    def test_boundary_abstraction_enforces_high_level_category_count(self):
        parsed = weighted_pool._parse_boundary_abstraction(
            json.dumps(
                {
                    "causal_categories": [
                        "sincere or literal intent without ironic reversal",
                        "insufficient cross-modal evidence of a contradiction",
                    ]
                }
            ),
            maximum_categories=2,
        )
        self.assertEqual(len(parsed["causal_categories"]), 2)
        with self.assertRaisesRegex(schema.SchemaError, "between 1 and 2"):
            weighted_pool._parse_boundary_abstraction(
                json.dumps({"causal_categories": ["one", "two", "three"]}),
                maximum_categories=2,
            )

    def test_boundary_compression_budget_uses_acceptance_tolerance_when_needed(self):
        self.assertEqual(
            weighted_pool._boundary_compression_word_budget(
                mechanism_words=39,
                profile_total_words=95,
                accepted_maximum_words=150,
                tolerance_margin_words=10,
            ),
            56,
        )
        self.assertEqual(
            weighted_pool._boundary_compression_word_budget(
                mechanism_words=102,
                profile_total_words=95,
                accepted_maximum_words=150,
                tolerance_margin_words=10,
            ),
            38,
        )
        with self.assertRaisesRegex(
            pool.ConsolidationError, "fewer than 5 words"
        ):
            weighted_pool._boundary_compression_word_budget(
                mechanism_words=146,
                profile_total_words=95,
                accepted_maximum_words=150,
                tolerance_margin_words=10,
            )

    def test_merge_enforces_separate_field_word_budgets(self):
        raw = json.dumps(
            {
                "mechanism": (
                    "Classify as sarcastic when praise contradicts visible failure "
                    "and thereby mocks the stated outcome."
                ),
                "exclusion_boundary": (
                    "Do not apply when praise and visible evidence consistently "
                    "support a sincere positive evaluation."
                ),
            }
        )
        with self.assertRaisesRegex(
            schema.SchemaError,
            r"mechanism has 14 words; maximum is 12",
        ):
            pool._parse_merge(
                raw,
                max_experience_words=40,
                max_mechanism_words=12,
                max_boundary_words=20,
            )

    def test_pool_rejects_concatenated_words(self):
        malformed = json.dumps(
            {
                "mechanism": (
                    "Classify as sarcastic when praise mocks an adverse result."
                ),
                "exclusion_boundary": (
                    "Do not apply when "
                    "alignmentexistswithoutexplicitpositivemarkersconfirmingsincereintent."
                ),
            }
        )
        with self.assertRaisesRegex(schema.SchemaError, "unnaturally long"):
            pool._parse_merge(malformed, max_experience_words=100)

    def test_boundary_merge_reports_remaining_boundary_word_limit(self):
        mechanism = "Classify as sarcastic when " + "signal " * 36 + "reverses meaning."
        boundary = "Do not apply when " + "literal " * 59 + "evidence remains."
        with self.assertRaisesRegex(
            schema.SchemaError,
            r"revised_exclusion_boundary has 65 words; maximum is 58",
        ):
            pool._parse_boundary_merge(
                json.dumps({"revised_exclusion_boundary": boundary}),
                mechanism=mechanism,
                max_experience_words=100,
            )

    def test_single_source_prompt_omits_unselected_sections(self):
        prompt = prompts.comparative_user_prompt(
            post_text="A post",
            teacher_reasoning=VALID_REASONING,
            student_reasoning=None,
            reference_reasoning=None,
            gold_label=None,
        )
        self.assertIn("TEACHER BLIND REASONING", prompt)
        self.assertNotIn("STUDENT BLIND REASONING", prompt)
        self.assertNotIn("GROUND-TRUTH REFERENCE REASONING", prompt)
        self.assertNotIn("GOLD LABEL", prompt)


class ReferenceMaterializationTests(unittest.TestCase):
    def test_aligned_reference_output_strips_audit_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            output_path = root / "aligned.jsonl"
            source_path.write_text(json.dumps({"sample_id": "sample-1", "analysis": VALID_REASONING}) + "\n", encoding="utf-8")
            image = root / "image.jpg"
            image.touch()
            sample = schema.SarcasmSample(
                sample_id="sample-1",
                image_path=str(image),
                text="A post",
                label="sarcastic",
            )
            cached_fingerprint = extraction._hash(
                {"sample_id": "sample-1", "gold_label": "sarcastic", "analysis": VALID_REASONING}
            )
            output_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "gold_label": "sarcastic",
                        "analysis": VALID_REASONING,
                        "origin": "old_cache",
                        "fingerprint": cached_fingerprint,
                        "prompt_versions": {"old": "audit-only"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = extraction._materialize_references(
                [sample], source_path=str(source_path), output_path=output_path
            )
            self.assertEqual(set(records["sample-1"]), {"sample_id", "gold_label", "analysis", "origin"})
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertNotIn("fingerprint", written)


class ReflectionTests(unittest.TestCase):
    def test_sentence_validation_handles_quoted_punctuation(self):
        content = (
            "A rhetorical question such as \u201cReally?\u201d can express skepticism when the surrounding evidence undermines a literal request. "
            "Classify it as sarcastic only when that skepticism targets the stated claim, not when the question sincerely requests clarification."
        )
        self.assertEqual(reflection._validate_english_sentences(content), content)
        curly_ending = (
            "When praise conflicts with an adverse result, verify that both concern the same target. "
            "Do not infer sarcasm when the praise remains literal.\u201d"
        )
        self.assertEqual(
            reflection._validate_english_sentences(curly_ending),
            curly_ending,
        )

    def test_sentence_validation_does_not_split_titles_or_initials(self):
        content = (
            "A report about Dr. Reddy remains literal when the visual evidence supports it. "
            "A list containing R. Rubio and S. Curry does not independently establish sarcasm."
        )
        self.assertEqual(reflection._validate_english_sentences(content), content)

    def test_extraction_repairs_mechanical_sentence_count_errors(self):
        one_sentence = json.dumps(
            {
                "experiences": [
                    {
                        "target_models": ["mllm_b"],
                        "content": (
                            "The factual wording matches the visible evidence, and no ironic reversal is supported."
                        ),
                    }
                ]
            }
        )
        parsed = extraction._parse_reflection_for_teacher_origin(
            one_sentence, "blind_correct"
        )
        self.assertEqual(
            parsed[0].content,
            "The factual wording matches the visible evidence. No ironic reversal is supported.",
        )

        four_sentences = json.dumps(
            {
                "experiences": [
                    {
                        "target_models": ["mllm_b"],
                        "content": (
                            "Check the literal claim. Compare it with the image. "
                            "Predict non-sarcastic when they align. Require a clear reversal for sarcasm."
                        ),
                    }
                ]
            }
        )
        parsed = extraction._parse_reflection_for_teacher_origin(
            four_sentences, "blind_correct"
        )
        self.assertEqual(len(reflection._split_english_sentences(parsed[0].content)), 3)

    def test_reflection_context_override_does_not_invalidate_cache(self):
        original = {"max_model_len": 12288, "reflection_max_new_tokens": 2048}
        overridden = {**original, "reflection_max_model_len": 16384}
        self.assertEqual(
            extraction._reflection_generation_fingerprint(overridden),
            original,
        )
        self.assertEqual(extraction._reflection_max_model_len(overridden), 16384)

    def test_document_compares_student_with_verified_teacher(self):
        document = reflection.embedded_reflection_prompt_document()
        self.assertEqual(document.path, "<embedded:pace_reflection>")
        self.assertTrue(document.sha256)
        sample = schema.SarcasmSample(
            sample_id="mmsd2:test",
            image_path="test-assets/not-used.jpg",
            text="Perfect weather.",
            label="sarcastic",
            source="mmsd2",
        )
        teacher = schema.validate_reasoning(VALID_REASONING)
        student = schema.validate_reasoning({**VALID_REASONING, "label": "non-sarcastic"})
        messages = reflection.build_comparative_reflection_messages(
            document, sample, teacher, student, "blind_correct"
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertEqual(messages[1]["content"][0], {"type": "image"})
        self.assertIn("<verified_teacher_response>", serialized)
        self.assertIn("<mllm_b_blind_response>", serialized)
        self.assertIn("<mllm_b_prediction_status>failed", serialized)
        self.assertNotIn("<ground_truth>", serialized)
        self.assertIn("<teacher_reasoning_origin>blind_correct", serialized)
        self.assertNotIn("<initial_teacher_blind_response>", serialized)
        self.assertNotIn("In PACE, MLLM A is the frozen teacher", serialized)
        self.assertNotIn("the teacher is not presumed correct", serialized)

    def test_documentation_prompt_matches_embedded_prompt(self):
        embedded = reflection.embedded_reflection_prompt_document()
        documented = reflection.load_reflection_prompt_document(
            ROOT / "prompts" / "comparative_reflection.md"
        )
        self.assertEqual(documented.common, embedded.common)
        self.assertEqual(documented.successful, embedded.successful)
        self.assertEqual(documented.failed, embedded.failed)
        self.assertEqual(documented.dataset_guidance, embedded.dataset_guidance)

    def test_correct_student_is_reviewed_only_for_missing_reasoning(self):
        document = reflection.embedded_reflection_prompt_document()
        sample = schema.SarcasmSample(
            sample_id="mmsd2:test",
            image_path="test-assets/not-used.jpg",
            text="Perfect weather.",
            label="sarcastic",
            source="mmsd2",
        )
        reasoning = schema.validate_reasoning(VALID_REASONING)
        messages = reflection.build_comparative_reflection_messages(
            document, sample, reasoning, reasoning, "blind_correct"
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertIn("<mllm_b_prediction_status>correct", serialized)
        self.assertIn("reasoning capabilities the student lacks", serialized)
        self.assertIn("reasoning is already equivalently supported", serialized)
        self.assertIn("exactly one transferable validation rule", serialized)
        self.assertIn("never return an empty array", serialized)

    def test_failed_teacher_compares_initial_with_verified_correction(self):
        document = reflection.embedded_reflection_prompt_document()
        sample = schema.SarcasmSample(
            sample_id="mmsd2:test",
            image_path="test-assets/not-used.jpg",
            text="Perfect weather.",
            label="sarcastic",
            source="mmsd2",
        )
        corrected = schema.validate_reasoning(VALID_REASONING)
        initial = schema.validate_reasoning(
            {**VALID_REASONING, "label": "non-sarcastic"}
        )
        messages = reflection.build_comparative_reflection_messages(
            document,
            sample,
            corrected,
            corrected,
            "codex_accepted",
            initial_teacher=initial,
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        user_text = messages[1]["content"][1]["text"]
        self.assertIn("<initial_teacher_blind_response>", serialized)
        self.assertIn('"label": "non-sarcastic"', user_text)
        self.assertIn("extract exactly one transferable corrective candidate", serialized)
        self.assertIn("source-agnostic detection rule", serialized)
        self.assertIn("downstream consumer", serialized)

    def test_failed_teacher_requires_initial_blind_reasoning(self):
        document = reflection.embedded_reflection_prompt_document()
        sample = schema.SarcasmSample(
            sample_id="mmsd2:test",
            image_path="test-assets/not-used.jpg",
            text="Perfect weather.",
            label="sarcastic",
            source="mmsd2",
        )
        reasoning = schema.validate_reasoning(VALID_REASONING)
        with self.assertRaisesRegex(
            schema.SchemaError, "requires its initial blind reasoning"
        ):
            reflection.build_comparative_reflection_messages(
                document,
                sample,
                reasoning,
                reasoning,
                "codex_revised",
            )

    def test_every_comparative_sample_rejects_empty_experience(self):
        for teacher_origin in ("blind_correct", "codex_accepted", "codex_revised"):
            with self.subTest(teacher_origin=teacher_origin):
                with self.assertRaisesRegex(
                    schema.SchemaError,
                    "array containing one to three items",
                ):
                    extraction._parse_reflection_for_teacher_origin(
                        '{"experiences": []}',
                        teacher_origin,
                    )

    def test_failed_teacher_empty_experience_gets_mandatory_retry_instruction(self):
        class Generator:
            def __init__(self):
                self.responses = [
                    ['{"experiences":[]}'],
                    [
                        json.dumps(
                            {
                                "experiences": [
                                    {
                                        "target_models": ["mllm_b"],
                                        "content": (
                                            "When apparently positive wording conflicts with concrete adverse evidence about the same target, verify whether the praise reverses the supported evaluation. "
                                            "Classify it as sarcastic only when that reversal communicates criticism, not when the wording remains sincere or the evidence concerns another target."
                                        ),
                                    }
                                ]
                            }
                        )
                    ],
                ]
                self.requests = []

            def generate_batch(self, requests):
                self.requests.append(requests)
                return self.responses.pop(0)

        sample = schema.SarcasmSample(
            sample_id="corrected-teacher",
            image_path="test-assets/not-used.jpg",
            text="Perfect weather.",
            label="sarcastic",
            source="mmsd2",
        )
        teacher = schema.validate_reasoning(VALID_REASONING)
        initial_teacher = schema.validate_reasoning(
            {**VALID_REASONING, "label": "non-sarcastic"}
        )
        generator = Generator()
        with tempfile.TemporaryDirectory() as directory:
            states = extraction._generate_reflection_batch(
                generator=generator,
                batch=[
                    (
                        sample,
                        teacher,
                        initial_teacher,
                        teacher,
                        "codex_accepted",
                        "fingerprint",
                    )
                ],
                mode="comparative",
                document=reflection.embedded_reflection_prompt_document(),
                failure_log_path=Path(directory) / "failures.jsonl",
                max_attempts=3,
                model_path="models/test-model",
            )

        self.assertEqual(len(states[0]["parsed"]), 1)
        retry_messages = generator.requests[1][0][0]
        self.assertEqual(
            retry_messages[-2],
            {
                "role": "assistant",
                "content": (
                    "I will discard the invalid content and return a fresh, corrected JSON object."
                ),
            },
        )
        self.assertIn(
            "Compare the initial incorrect teacher reasoning with the verified corrected reasoning",
            retry_messages[-1]["content"],
        )
        self.assertIn("an empty array is forbidden", retry_messages[-1]["content"])

    def test_non_english_retry_does_not_reinject_invalid_content(self):
        class Generator:
            def __init__(self):
                self.responses = [
                    [
                        json.dumps(
                            {
                                "experiences": [
                                    {
                                        "target_models": ["mllm_b"],
                                        "content": "Quoted Chinese text: 欲盖弥彰. This supports sarcasm.",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    ],
                    [
                        json.dumps(
                            {
                                "experiences": [
                                    {
                                        "target_models": ["mllm_b"],
                                        "content": (
                                            "A rhetorical question can challenge a literal claim when surrounding evidence exposes a contradiction. "
                                            "This supports sarcasm only when the challenge reverses the apparent meaning, not when it requests information."
                                        ),
                                    }
                                ]
                            }
                        )
                    ],
                ]
                self.requests = []

            def generate_batch(self, requests):
                self.requests.append(requests)
                return self.responses.pop(0)

        sample = schema.SarcasmSample(
            sample_id="non-english-retry",
            image_path="test-assets/not-used.jpg",
            text="A post",
            label="sarcastic",
            source="mmsd2",
        )
        reasoning = schema.validate_reasoning(VALID_REASONING)
        generator = Generator()
        with tempfile.TemporaryDirectory() as directory:
            states = extraction._generate_reflection_batch(
                generator=generator,
                batch=[
                    (
                        sample,
                        reasoning,
                        reasoning,
                        reasoning,
                        "blind_correct",
                        "fingerprint",
                    )
                ],
                mode="comparative",
                document=reflection.embedded_reflection_prompt_document(),
                failure_log_path=Path(directory) / "failures.jsonl",
                max_attempts=3,
                model_path="models/test-model",
            )

        self.assertEqual(len(states[0]["parsed"]), 1)
        retry_messages = generator.requests[1][0][0]
        self.assertNotIn("欲盖弥彰", retry_messages[-2]["content"])
        self.assertIn("ASCII characters only", retry_messages[-1]["content"])

    def test_v12_nonempty_cache_identity_is_available_for_targeted_backfill(self):
        self.assertIn(
            (
                "pace-comparative-reflection-v12-shared-teacher-student-errors",
                "5a1ff5f158a9fbdd3b2486427333abd6e303314965194ae8c30ec64b56d7a9cf",
            ),
            reflection.LEGACY_REFLECTION_CACHE_IDENTITIES,
        )
        self.assertEqual(
            reflection.PROMPT_VERSION,
            "pace-comparative-reflection-v13-minimum-one-experience",
        )
        self.assertIn(
            "minItems",
            extraction.REFLECTION_JSON_SCHEMA["properties"]["experiences"],
        )
        self.assertNotIn(
            "minItems",
            extraction._legacy_reflection_schema()["properties"]["experiences"],
        )

    def test_incomplete_extraction_summary_is_fatal(self):
        with self.assertRaisesRegex(
            extraction.ExtractionError,
            "experience_invalid=2, invalid_pairs=0",
        ):
            extraction._require_complete_extraction(
                {"experience_invalid": 2, "invalid_pairs": 0}
            )
        extraction._require_complete_extraction(
            {"experience_invalid": 0, "invalid_pairs": 0}
        )
        with self.assertRaisesRegex(
            extraction.ExtractionError,
            "experience_null=1",
        ):
            extraction._require_complete_extraction(
                {
                    "experience_invalid": 0,
                    "invalid_pairs": 0,
                    "experience_null": 1,
                }
            )

    def test_reflection_rejects_non_student_target(self):
        raw = json.dumps(
            {
                "experiences": [
                    {
                        "target_models": ["mllm_a"],
                        "content": (
                            "Check the decisive evidence relationship before classifying. "
                            "Use the supported direction unless the nearest boundary applies."
                        ),
                    }
                ]
            }
        )
        with self.assertRaises(schema.SchemaError):
            reflection.parse_reflection(raw)

    def test_teacher_label_correction_prompt_contains_gold_and_initial_trace(self):
        messages = prompts.teacher_label_correction_messages(
            "A post",
            "sarcastic",
            VALID_REASONING,
        )
        serialized = json.dumps(messages)
        self.assertIn("<gold_label>sarcastic</gold_label>", serialized)
        self.assertIn("<initial_teacher_reasoning>", serialized)
        self.assertIn("Correction mode", serialized)


class TeacherCorrectionReviewTests(unittest.TestCase):
    def test_failed_teacher_requires_codex_review_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.touch()
            correct = schema.SarcasmSample(
                sample_id="correct",
                image_path=str(image),
                text="Perfect weather.",
                label="sarcastic",
            )
            failed = schema.SarcasmSample(
                sample_id="failed",
                image_path=str(image),
                text="Perfect weather.",
                label="sarcastic",
            )
            failed_blind = {**VALID_REASONING, "label": "non-sarcastic"}
            records = {
                "correct": {"sample_id": "correct", "analysis": VALID_REASONING},
                "failed": {"sample_id": "failed", "analysis": failed_blind},
            }
            corrected = {
                "sample_id": "failed",
                "analysis": VALID_REASONING,
                "fingerprint": "correction-v1",
            }
            original = extraction._generate_reasonings
            extraction._generate_reasonings = lambda **_kwargs: {"failed": corrected}
            config = {
                "models": {"teacher": {"path": "models/fake-teacher"}},
                "output": {"root": str(root)},
            }
            try:
                verified, status = extraction._prepare_verified_teacher_reasonings(
                    config,
                    [correct, failed],
                    records,
                    output_dir=root,
                    generation={},
                    failure_log_path=root / "failures.jsonl",
                    json_max_attempts=3,
                )
                self.assertEqual(set(verified), {"correct"})
                self.assertEqual(status["teacher_review_pending"], 1)
                self.assertIs(status["teacher_review_queue_ready"], True)
                review = {
                    "sample_id": "failed",
                    "correction_fingerprint": "correction-v1",
                    "decision": "ACCEPT",
                    "revised_reasoning": None,
                    "rationale": "The corrected trace supports the supplied label.",
                }
                (root / "teacher_correction_reviews.jsonl").write_text(
                    json.dumps(review) + "\n", encoding="utf-8"
                )
                verified, status = extraction._prepare_verified_teacher_reasonings(
                    config,
                    [correct, failed],
                    records,
                    output_dir=root,
                    generation={},
                    failure_log_path=root / "failures.jsonl",
                    json_max_attempts=3,
                )
            finally:
                extraction._generate_reasonings = original
            self.assertEqual(set(verified), {"correct", "failed"})
            self.assertEqual(verified["failed"]["origin"], "codex_accepted")
            self.assertEqual(status["teacher_review_pending"], 0)

    def test_stale_codex_review_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.touch()
            sample = schema.SarcasmSample(
                sample_id="failed",
                image_path=str(image),
                text="Perfect weather.",
                label="sarcastic",
            )
            records = {
                "failed": {
                    "sample_id": "failed",
                    "analysis": {**VALID_REASONING, "label": "non-sarcastic"},
                }
            }
            corrected = {
                "sample_id": "failed",
                "analysis": VALID_REASONING,
                "fingerprint": "correction-v2",
            }
            review = {
                "sample_id": "failed",
                "correction_fingerprint": "correction-v1",
                "decision": "ACCEPT",
                "revised_reasoning": None,
                "rationale": "This reviewed an older correction.",
            }
            (root / "teacher_correction_reviews.jsonl").write_text(
                json.dumps(review) + "\n", encoding="utf-8"
            )
            original = extraction._generate_reasonings
            extraction._generate_reasonings = lambda **_kwargs: {"failed": corrected}
            try:
                verified, status = extraction._prepare_verified_teacher_reasonings(
                    {"models": {"teacher": {"path": "models/fake-teacher"}}},
                    [sample],
                    records,
                    output_dir=root,
                    generation={},
                    failure_log_path=root / "failures.jsonl",
                    json_max_attempts=3,
                )
            finally:
                extraction._generate_reasonings = original
            self.assertEqual(verified, {})
            self.assertEqual(status["teacher_review_pending"], 1)
            queue = json.loads(
                (root / "teacher_correction_review_queue.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(queue["review_status"], "STALE")
            self.assertEqual(queue["correction_fingerprint"], "correction-v2")

    def test_review_queue_waits_for_every_teacher_reasoning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.touch()
            ready = schema.SarcasmSample(
                sample_id="ready",
                image_path=str(image),
                text="A post.",
                label="sarcastic",
            )
            missing = schema.SarcasmSample(
                sample_id="missing",
                image_path=str(image),
                text="Another post.",
                label="sarcastic",
            )
            original = extraction._generate_reasonings
            extraction._generate_reasonings = lambda **_kwargs: {}
            try:
                _verified, status = extraction._prepare_verified_teacher_reasonings(
                    {"models": {"teacher": {"path": "models/fake-teacher"}}},
                    [ready, missing],
                    {"ready": {"sample_id": "ready", "analysis": VALID_REASONING}},
                    output_dir=root,
                    generation={},
                    failure_log_path=root / "failures.jsonl",
                    json_max_attempts=3,
                )
            finally:
                extraction._generate_reasonings = original
            self.assertIs(status["teacher_review_queue_ready"], False)
            self.assertEqual(
                (root / "teacher_correction_review_queue.jsonl").read_text(
                    encoding="utf-8"
                ),
                "",
            )

    def test_reflection_output_schema_is_strict(self):
        raw = json.dumps(
            {
                "experiences": [
                    {
                        "target_models": ["mllm_b"],
                        "content": (
                            "When literal praise conflicts with a visibly adverse outcome, verify "
                            "that both modalities concern the same target. This supports sarcasm only "
                            "when the praise pragmatically criticizes that outcome, not when the image "
                            "is merely unrelated."
                        ),
                    }
                ]
            }
        )
        parsed = reflection.parse_reflection(raw)
        self.assertEqual(parsed[0].target_models, ("mllm_b",))
        with self.assertRaises(schema.SchemaError):
            reflection.parse_reflection('{"experience":"old contract"}')


class WeightedReferenceTests(unittest.TestCase):
    def test_replay_rejects_legacy_semantic_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actions_path = root / "reference_actions.jsonl"
            actions_path.write_text(
                json.dumps(
                    {
                        "sample_id": "legacy-candidate",
                        "method": weighted_pool.REFERENCE_FLOW_METHOD,
                        "calls": {
                            "merge_compression": {
                                "fallback_action": "UPVOTE"
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                weighted_pool.ConsolidationError,
                r"reference action 1 \(legacy-candidate\)",
            ):
                weighted_pool._replay_state(
                    [
                        {
                            "sample_id": "legacy-candidate",
                            "gold_label": "sarcastic",
                        }
                    ],
                    actions_path,
                    root / "reference_evaluations.jsonl",
                )

    def test_group_candidate_path_filters_the_selected_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extraction_dir = root / "experience_extraction" / "selected_group"
            extraction_dir.mkdir(parents=True)
            comparative_records = [
                {
                    "sample_id": sample_id,
                    "gold_label": label,
                    "experiences": [
                        {
                            "target_models": ["mllm_b"],
                            "content": f"Candidate for {sample_id}. A boundary follows.",
                        }
                    ],
                    "error": None,
                }
                for sample_id, label in (
                    ("included", "sarcastic"),
                    ("excluded", "non-sarcastic"),
                )
            ]
            (extraction_dir / "comparative_experiences.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in comparative_records),
                encoding="utf-8",
            )
            group_path = root / "selected_group.jsonl"
            group_path.write_text(
                json.dumps({"sample_id": "included"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "experience_extraction": {
                    "output_path": "experience_extraction/selected_group"
                },
                "experience_pool": {
                    "path": "experience_pools/selected_group/experience_pool.json",
                    "candidate_sample_path": "selected_group.jsonl",
                },
                "output": {"root": str(root)},
            }

            paths, candidates, total = weighted_pool._segment_paths_and_candidates(
                config,
                None,
            )

            self.assertEqual(total, 1)
            self.assertEqual(
                [candidate["sample_id"] for candidate in candidates],
                ["included::item-00"],
            )
            self.assertEqual(
                paths["pool"],
                root / "experience_pools" / "selected_group" / "experience_pool.json",
            )

    def test_alarm_operations_update_importance_and_boundary_lock(self):
        mechanism = (
            "Classify as sarcastic when literal praise pragmatically mocks "
            "a clearly adverse result."
        )
        boundary = (
            "Do not apply when the praise is sincere or the result is unrelated."
        )
        items = []
        weighted_pool._apply_structural_action(
            items,
            {
                "action": "ADD",
                "target_id": None,
                "experience": f"{mechanism} {boundary}",
                "mechanism": mechanism,
                "exclusion_boundary": boundary,
                "modification_scope": "FULL",
                "assigned_id": "exp-000001",
                "rationale": "New mechanism.",
            },
        )
        weighted_pool._apply_structural_action(
            items,
            {
                "action": "UPVOTE",
                "target_id": "exp-000001",
                "rationale": "The candidate supports this rule.",
            },
        )
        revised_boundary = (
            "Do not apply when the praise is sincere, the result is unrelated, "
            "or the apparent contrast has no evaluative target."
        )
        weighted_pool._apply_structural_action(
            items,
            {
                "action": "MODIFY",
                "target_id": "exp-000001",
                "experience": f"{mechanism} {revised_boundary}",
                "mechanism": mechanism,
                "exclusion_boundary": revised_boundary,
                "modification_scope": "BOUNDARY",
                "assigned_id": None,
                "rationale": "Adds a supported exclusion boundary.",
            },
        )
        self.assertEqual(items[0]["mechanism"], mechanism)
        self.assertEqual(items[0]["exclusion_boundary"], revised_boundary)
        self.assertEqual(items[0]["boundary_modify_count"], 1)

        with self.assertRaisesRegex(
            pool.ConsolidationError, "attempted to change the target mechanism"
        ):
            weighted_pool._apply_structural_action(
                items,
                {
                    "action": "MODIFY",
                    "target_id": "exp-000001",
                    "experience": f"Classify as sarcastic when altered. {revised_boundary}",
                    "mechanism": "Classify as sarcastic when altered.",
                    "exclusion_boundary": revised_boundary,
                    "modification_scope": "BOUNDARY",
                    "assigned_id": None,
                    "rationale": "Invalid mechanism rewrite.",
                },
            )

        weighted_pool._apply_structural_action(
            items,
            {
                "action": "DOWNVOTE",
                "target_id": "exp-000001",
                "rationale": "Known-label evidence directly contradicts it.",
            },
        )
        self.assertEqual(items[0]["importance"], 3)
        self.assertEqual(items[0]["revision"], 2)
        self.assertEqual(items[0]["upvote_count"], 1)
        self.assertEqual(items[0]["modify_count"], 1)
        self.assertEqual(items[0]["downvote_count"], 1)
        for _ in range(3):
            weighted_pool._apply_structural_action(
                items,
                {
                    "action": "DOWNVOTE",
                    "target_id": "exp-000001",
                    "rationale": "Further direct contradiction.",
                },
            )
        self.assertFalse(items[0]["active"])
        self.assertEqual(items[0]["importance"], 0)

    def test_label_specific_relation_schema_and_target_validation(self):
        raw = json.dumps(
            {
                "action": "MODIFY",
                "target_id": "exp-000014",
                "rationale": "The candidate adds a valid boundary.",
            }
        )
        with self.assertRaisesRegex(
            schema.SchemaError,
            "exp-000001, exp-000003",
        ):
            weighted_pool._parse_weighted_relation(
                raw,
                {"exp-000001", "exp-000003"},
                "sarcastic",
            )

        relation_schema = weighted_pool._weighted_relation_schema(
            {"exp-000003", "exp-000001"},
            "non-sarcastic",
        )
        self.assertEqual(
            set(relation_schema["properties"]["action"]["enum"]),
            {"MODIFY", "UPVOTE", "DOWNVOTE", "SKIP"},
        )
        with self.assertRaisesRegex(schema.SchemaError, "non-sarcastic"):
            weighted_pool._parse_weighted_relation(
                json.dumps(
                    {
                        "action": "ADD",
                        "target_id": None,
                        "rationale": "Invalid standalone negative rule.",
                    }
                ),
                {"exp-000001"},
                "non-sarcastic",
            )

    def test_reference_limit_keeps_high_weight_and_newer_ties(self):
        items = [
            weighted_pool._new_reference(
                f"exp-{index:06d}",
                f"Classify as sarcastic when mechanism {index} is observable.",
                "Do not apply when the apparent signal lacks mocking intent.",
                index,
            )
            for index in range(1, 4)
        ]
        deactivated = weighted_pool._enforce_reference_limit(items, 2)
        self.assertEqual(deactivated, ["exp-000001"])
        self.assertEqual(
            [item["id"] for item in weighted_pool._ordered_active(items)],
            ["exp-000003", "exp-000002"],
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pool.json"
            weighted_pool._export_pool(output, items)
            experiences = json.loads(output.read_text(encoding="utf-8"))["experiences"]
        self.assertEqual(
            [item["id"] for item in experiences],
            ["exp-000003", "exp-000002", "exp-000001"],
        )
        self.assertTrue(
            all(set(item) == {"id", "text", "active"} for item in experiences)
        )

    def test_fresh_refine_archives_legacy_outputs_and_normalizations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_dir = root / "experience_pool"
            pool_dir.mkdir()
            pool_path = pool_dir / "experience_pool.json"
            pool_path.write_text('{"experiences": []}\n', encoding="utf-8")
            dedup_path = pool_dir / "dedup_checkpoint_005902_candidates.jsonl"
            dedup_path.write_text("{}\n", encoding="utf-8")
            legacy_evaluations = pool_dir / "reference_evaluations.jsonl"
            legacy_evaluations.write_text("{}\n", encoding="utf-8")
            normalizations = pool_dir / "consolidation_normalizations.jsonl"
            normalizations.write_text("{}\n", encoding="utf-8")
            paths = weighted_pool._operational_paths(
                {
                    "experience_pool": {"path": str(pool_path)},
                    "output": {"root": str(root)},
                }
            )
            archive = weighted_pool._archive_operational_outputs(
                paths,
                include_normalizations=True,
            )

            self.assertIsNotNone(archive)
            self.assertTrue((archive / dedup_path.name).is_file())
            self.assertTrue((archive / legacy_evaluations.name).is_file())
            self.assertTrue((archive / normalizations.name).is_file())
            self.assertFalse(dedup_path.exists())
            self.assertFalse(legacy_evaluations.exists())
            self.assertFalse(normalizations.exists())

    def test_replay_preserves_label_aware_votes_and_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [
                {"sample_id": "c1", "gold_label": "sarcastic"},
                {"sample_id": "c2", "gold_label": "sarcastic"},
                {"sample_id": "c3", "gold_label": "sarcastic"},
                {"sample_id": "c4", "gold_label": "non-sarcastic"},
            ]
            mechanism = (
                "Classify as sarcastic when literal praise pragmatically mocks "
                "a clearly adverse result."
            )
            boundary = (
                "Do not apply when the praise is sincere or the result is unrelated."
            )
            revised_boundary = (
                "Do not apply when the praise is sincere, the result is unrelated, "
                "or the contrast has no evaluative target."
            )
            actions = [
                {
                    "sample_id": "c1",
                    "method": weighted_pool.REFERENCE_FLOW_METHOD,
                    "candidate_label": "sarcastic",
                    "action": "ADD",
                    "target_id": None,
                    "experience": f"{mechanism} {boundary}",
                    "mechanism": mechanism,
                    "exclusion_boundary": boundary,
                    "modification_scope": "FULL",
                    "assigned_id": "exp-000001",
                    "rationale": "New.",
                },
                {
                    "sample_id": "c2",
                    "method": weighted_pool.REFERENCE_FLOW_METHOD,
                    "candidate_label": "sarcastic",
                    "action": "UPVOTE",
                    "target_id": "exp-000001",
                    "rationale": "Covered.",
                },
                {
                    "sample_id": "c3",
                    "method": weighted_pool.REFERENCE_FLOW_METHOD,
                    "candidate_label": "sarcastic",
                    "action": "MODIFY",
                    "target_id": "exp-000001",
                    "experience": f"{mechanism} {revised_boundary}",
                    "mechanism": mechanism,
                    "exclusion_boundary": revised_boundary,
                    "modification_scope": "FULL",
                    "assigned_id": None,
                    "rationale": "Stronger.",
                },
                {
                    "sample_id": "c4",
                    "method": weighted_pool.REFERENCE_FLOW_METHOD,
                    "candidate_label": "non-sarcastic",
                    "action": "DOWNVOTE",
                    "target_id": "exp-000001",
                    "rationale": "Directly contradicted.",
                },
            ]
            action_path = root / "actions.jsonl"
            action_path.write_text(
                "".join(json.dumps(value) + "\n" for value in actions),
                encoding="utf-8",
            )
            evaluation_path = root / "legacy_evaluations.jsonl"
            items, processed, counts, next_id, legacy_final = weighted_pool._replay_state(
                candidates,
                action_path,
                evaluation_path,
            )

            self.assertEqual(processed, 4)
            self.assertEqual(next_id, 2)
            self.assertFalse(legacy_final)
            self.assertEqual(counts["UPVOTE"], 1)
            self.assertEqual(counts["DOWNVOTE"], 1)
            self.assertEqual(items[0]["importance"], 3)
            self.assertEqual(items[0]["revision"], 2)
            self.assertEqual(items[0]["mechanism"], mechanism)
            self.assertEqual(items[0]["exclusion_boundary"], revised_boundary)

            evaluation_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                pool.ConsolidationError, "legacy per-reference evaluation"
            ):
                weighted_pool._replay_state(
                    candidates,
                    action_path,
                    evaluation_path,
                )

    def test_weighted_pipeline_builds_boundary_rules_and_resumes(self):
        class ConsolidationGenerator:
            def __init__(self):
                self.max_new_tokens = 32
                self.chat_template_kwargs = {"enable_thinking": False}
                self.json_schema = None
                self.last_num_cached_tokens = 0
                self.total_num_cached_tokens = 0

            @staticmethod
            def count_input_tokens(messages):
                return len(json.dumps(messages))

            def generate(self, messages, *, image_path):
                if image_path is not None:
                    raise AssertionError("reference refinement must remain text-only")
                system = messages[0]["content"]
                payload = json.loads(messages[1]["content"])
                if system == prompts.CANDIDATE_NORMALIZATION_SYSTEM_PROMPT:
                    if payload["gold_label"] == "non-sarcastic":
                        return json.dumps(
                            {
                                "valid": True,
                                "mechanism": None,
                                "exclusion_boundary": (
                                    "Do not infer sarcasm when an apparent contrast "
                                    "has no evaluative target or mocking intent."
                                ),
                                "rationale": "A transferable countercondition remains.",
                            }
                        )
                    name = payload["candidate_experience"].split()[1].lower()
                    return json.dumps(
                        {
                            "valid": True,
                            "mechanism": (
                                "Classify as sarcastic when transferable "
                                f"{name} evidence pragmatically supports mockery."
                            ),
                            "exclusion_boundary": (
                                "Do not apply when the apparent signal is literal "
                                "or lacks a mocking target."
                            ),
                            "rationale": "A transferable sarcasm mechanism remains.",
                        }
                    )
                if system == prompts.CONSOLIDATION_SYSTEM_PROMPT:
                    candidate = payload["candidate"]
                    if candidate["gold_label"] == "non-sarcastic":
                        return json.dumps(
                            {
                                "action": "MODIFY",
                                "target_id": "exp-000002",
                                "rationale": "The counterexample adds a missing boundary.",
                            }
                        )
                    mechanism = candidate["mechanism"].lower()
                    if "four" in mechanism:
                        return json.dumps(
                            {
                                "action": "MODIFY",
                                "target_id": "exp-000002",
                                "rationale": "The candidate strengthens this mechanism.",
                            }
                        )
                    if "three" in mechanism:
                        return json.dumps(
                            {
                                "action": "UPVOTE",
                                "target_id": "exp-000002",
                                "rationale": "The second rule fully covers it.",
                            }
                        )
                    return json.dumps(
                        {
                            "action": "ADD",
                            "target_id": None,
                            "rationale": "This is a new sarcasm mechanism.",
                        }
                    )
                if system == prompts.EXPERIENCE_MERGE_SYSTEM_PROMPT:
                    return json.dumps(
                        {
                            "mechanism": (
                                "Classify as sarcastic when "
                                + "observable evidence " * 30
                                + "supports mockery."
                            ),
                            "exclusion_boundary": (
                                "Do not apply when "
                                + "literal context " * 25
                                + "removes mocking intent."
                            ),
                        }
                    )
                if system.startswith(
                    "You independently synthesize one compact multimodal sarcasm-detection experience"
                ):
                    target_words = payload["target_maximum_words"]
                    if (
                        payload["maximum_mechanism_words"]
                        + payload["maximum_exclusion_boundary_words"]
                        != target_words
                    ):
                        raise AssertionError("field budgets do not sum to target")
                    if target_words == 95 and len(messages) == 2:
                        return json.dumps(
                            {
                                "mechanism": (
                                    "Classify as sarcastic when "
                                    + "observable evidence " * 30
                                    + "supports mockery."
                                ),
                                "exclusion_boundary": (
                                    "Do not apply when "
                                    + "literal context " * 25
                                    + "removes mocking intent."
                                ),
                            }
                        )
                    if target_words == 95 and len(messages) > 2:
                        return json.dumps(
                            {
                                "mechanism": (
                                    "Classify as sarcastic when an observable "
                                    "contrast pragmatically mocks its target."
                                ),
                                "exclusion_boundary": (
                                    "Do not apply when the contrast is incidental "
                                    "or lacks mocking intent."
                                ),
                            }
                        )
                    raise AssertionError("unexpected compression request")
                if system == prompts.EXCLUSION_BOUNDARY_MERGE_SYSTEM_PROMPT:
                    self.assert_boundary_payload(payload)
                    return json.dumps(
                        {
                            "revised_exclusion_boundary": (
                                "Do not apply when "
                                + "literal context " * 55
                                + "or sincere intent resolves the contrast."
                            )
                        }
                    )
                if system.startswith(
                    "You plan a semantic resynthesis of an overlong exclusion boundary"
                ):
                    return json.dumps(
                        {
                            "causal_categories": [
                                "literal context or sincere intent resolves the contrast",
                                "the evidence lacks an evaluative target",
                            ]
                        }
                    )
                if system.startswith(
                    "You independently synthesize one compact exclusion boundary"
                ):
                    return json.dumps(
                        {
                            "revised_exclusion_boundary": (
                                "Do not apply when the contrast is incidental, "
                                "lacks an evaluative target, or lacks mocking intent."
                            )
                        }
                    )
                raise AssertionError("unexpected reference-refinement prompt")

            @staticmethod
            def assert_boundary_payload(payload):
                if not payload["target_mechanism"].startswith(
                    "Classify as sarcastic when "
                ):
                    raise AssertionError("boundary merge did not preserve mechanism")
                if not payload["candidate_boundary_evidence"].startswith(
                    "Do not infer sarcasm when "
                ):
                    raise AssertionError("boundary evidence has the wrong direction")

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extraction_dir = root / "experience_extraction"
            extraction_dir.mkdir()
            records = [
                {
                    "sample_id": "positive",
                    "gold_label": "sarcastic",
                    "experiences": [
                        {
                            "target_models": ["mllm_b"],
                            "content": f"Candidate {name}. A boundary.",
                        }
                        for name in ("one", "two", "three", "four")
                    ],
                },
                {
                    "sample_id": "negative",
                    "gold_label": "non-sarcastic",
                    "experiences": [
                        {
                            "target_models": ["mllm_b"],
                            "content": "Candidate counterexample. A boundary.",
                        }
                    ],
                },
            ]
            (extraction_dir / "comparative_experiences.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            config = {
                "models": {"teacher": {"path": "models/fake-teacher"}},
                "experience_extraction": {"output_path": "experience_extraction"},
                "experience_pool": {
                    "path": "experience_pool/experience_pool.json",
                    "references": {"target_size": 2, "isolate_stages": False},
                    "generation": {
                        "deterministic": True,
                        "max_input_tokens": 100000,
                        "max_experience_words": 100,
                        "show_progress": False,
                        "pipeline": {"enabled": False},
                        "normalization": {
                            "max_new_tokens": 32,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        "comparison": {
                            "max_new_tokens": 32,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        "merge": {
                            "max_new_tokens": 32,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        "json": {
                            "max_attempts": 3,
                            "failure_log_path": "failures.jsonl",
                        },
                    },
                },
                "output": {"root": str(root)},
            }
            original_builder = weighted_pool._build_consolidation_generator
            try:
                weighted_pool._build_consolidation_generator = (
                    lambda *_args, **_kwargs: ConsolidationGenerator()
                )
                result = weighted_pool.run_weighted_reference_pipeline(config)
            finally:
                weighted_pool._build_consolidation_generator = original_builder

            resumed = weighted_pool.run_weighted_reference_pipeline(config)
            actions = list(
                schema.load_records_by_id(
                    root / "experience_pool/reference_actions.jsonl"
                ).values()
            )
            normalizations = list(
                schema.load_records_by_id(
                    root / "experience_pool/consolidation_normalizations.jsonl"
                ).values()
            )
            exported = json.loads(
                (root / "experience_pool/experience_pool.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(result["status"], "consolidation_complete")
            self.assertEqual(resumed["processed"], 5)
            self.assertEqual(
                [value["action"] for value in actions],
                ["ADD", "ADD", "UPVOTE", "MODIFY", "MODIFY"],
            )
            self.assertEqual(actions[-2]["modification_scope"], "FULL")
            self.assertEqual(actions[-1]["modification_scope"], "BOUNDARY")
            self.assertEqual(actions[-1]["mechanism"], actions[-2]["mechanism"])
            compression_call = actions[-2]["calls"]["merge_compression"]
            self.assertEqual(compression_call["attempts"], 2)
            self.assertEqual(compression_call["target_word_limits"], [95])
            boundary_compression_call = actions[-1]["calls"][
                "boundary_compression"
            ]
            self.assertEqual(boundary_compression_call["attempts"], 2)
            self.assertEqual(boundary_compression_call["target_word_limits"], [95])
            self.assertEqual(
                boundary_compression_call["abstractions"][0][
                    "maximum_categories"
                ],
                4,
            )
            self.assertTrue(
                all(
                    value["method"] == weighted_pool.REFERENCE_FLOW_METHOD
                    for value in actions
                )
            )
            self.assertEqual(
                {value["gold_label"] for value in normalizations},
                {"sarcastic", "non-sarcastic"},
            )
            self.assertTrue(
                all(
                    value["method"] == pool.NORMALIZATION_METHOD
                    for value in normalizations
                )
            )
            self.assertFalse(
                (root / "experience_pool/reference_evaluations.jsonl").exists()
            )
            self.assertTrue(
                all(
                    set(item) == {"id", "text", "active"}
                    for item in exported["experiences"]
                )
            )
            active_texts = [
                item["text"] for item in exported["experiences"] if item["active"]
            ]
            self.assertTrue(
                all(
                    text.startswith("Classify as sarcastic when ")
                    and " Do not apply when " in text
                    for text in active_texts
                )
            )
            rendered = pool.load_active_experience_text(
                root / "experience_pool/experience_pool.json"
            )
            self.assertNotIn("Classify as non-sarcastic", rendered)
            self.assertNotIn("Do not infer sarcasm when", rendered)

    def test_isolated_stage_propagates_child_failure(self):
        with self.assertRaisesRegex(
            pool.ConsolidationError,
            "unknown isolated reference stage",
        ):
            weighted_pool._run_isolated_stage("invalid", {}, None)


class PreflightTests(unittest.TestCase):
    def test_vllm_reasoning_requires_thinking_mode(self):
        with self.assertRaisesRegex(modeling.ModelLoadError, "enable_thinking=true"):
            modeling.VllmMultimodalGenerator(
                "models/missing",
                role="student",
                max_new_tokens=1536,
                deterministic=True,
                chat_template_kwargs={"enable_thinking": False},
                max_model_len=6144,
                max_num_seqs=16,
                gpu_memory_utilization=0.75,
                thinking_token_budget=None,
                json_schema=schema.REASONING_JSON_SCHEMA,
                enforce_eager=False,
            )

    def test_local_model_pair_is_compatible_and_multimodal(self):
        teacher_model = Path(
            os.environ.get("PACE_TEACHER_MODEL", "models/Qwen3.5-9B")
        )
        student_model = Path(
            os.environ.get("PACE_STUDENT_MODEL", "models/Qwen3.5-4B")
        )
        if not teacher_model.is_absolute():
            teacher_model = ROOT / teacher_model
        if not student_model.is_absolute():
            student_model = ROOT / student_model
        if not teacher_model.is_dir() or not student_model.is_dir():
            self.skipTest(
                "set PACE_TEACHER_MODEL and PACE_STUDENT_MODEL "
                "to local multimodal model directories"
            )
        result = preflight.validate_model_pair(
            str(teacher_model),
            str(student_model),
        )
        self.assertTrue(result["compatible"])
        self.assertEqual(result["teacher"]["architecture"], "Qwen3_5ForConditionalGeneration")
        self.assertEqual(result["student"]["vocab_size"], 248320)
        self.assertEqual(
            result["teacher"]["input_contract_hashes"],
            result["student"]["input_contract_hashes"],
        )

    def test_qwen35_cross_device_placement_is_rejected(self):
        class Model:
            hf_device_map = {"model.layers.0": 0, "model.layers.1": 1}

        with self.assertRaisesRegex(modeling.ModelLoadError, "Set generation.device_map"):
            modeling._assert_supported_device_placement(
                Model(), {"model_type": "qwen3_5"}, role="teacher"
            )

    def test_cuda_cache_cleanup_never_masks_generation_failure(self):
        class FailingCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                raise RuntimeError("poisoned CUDA context")

        torch = type("Torch", (), {"cuda": FailingCuda})()
        modeling._empty_cuda_cache_safely(torch)



class MetricTests(unittest.TestCase):
    def test_metrics_report_macro_f1_and_invalid_json(self):
        result = metrics.classification_metrics(
            ["sarcastic", "sarcastic", "non-sarcastic"],
            ["sarcastic", "non-sarcastic", "non-sarcastic"],
            invalid=1,
        )
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["invalid_json"], 1)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertIn("macro_f1", result)

    def test_invalid_predictions_reduce_class_recall(self):
        result = metrics.classification_metrics(
            ["non-sarcastic"],
            ["non-sarcastic"],
            invalid=1,
            invalid_gold=["sarcastic"],
        )
        self.assertEqual(result["per_class"]["sarcastic"]["support"], 1)
        self.assertEqual(result["per_class"]["sarcastic"]["recall"], 0.0)


class StageBSourceTests(unittest.TestCase):
    def test_direct_distillation_task_reward_is_zero(self):
        self.assertEqual(
            reward.compute_score("pace_msd", "prediction", "sarcastic"),
            0.0,
        )

    def test_stage_b_invariants_are_wired_to_native_distillation(self):
        adapter = (TRAINER / "pace_agent_loop.py").read_text(encoding="utf-8")
        cli = (ROOT / "tools" / "pace_cli.py").read_text(encoding="utf-8")
        main_ppo = (TRAINER / "main_ppo.py").read_text(encoding="utf-8")
        config = (ROOT / "configs" / "pace_msd.yaml").read_text(encoding="utf-8")

        self.assertIn("experience_conditioned_messages", adapter)
        self.assertIn("teacher must score the exact student-generated response token IDs", adapter)
        self.assertIn("aligned_logprobs[target_start:target_end]", adapter)
        self.assertIn("PaceAgentLoopManager", adapter)
        self.assertIn("distillation.enabled=True", cli)
        self.assertIn("data.dataloader_num_workers=0", cli)
        self.assertIn("data.filter_overlong_prompts=False", cli)
        self.assertNotIn("data.filter_overlong_prompts=True", cli)
        self.assertIn('_model_context_limit(get_required(config, "models.student.path"))', cli)
        self.assertIn('_model_context_limit(get_required(config, "models.teacher.path"))', cli)
        self.assertIn("actor_rollout_ref.actor.use_dynamic_bsz=False", cli)
        self.assertIn("override_optimizer_config={foreach:false}", cli)
        self.assertIn("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1", cli)
        self.assertIn(
            "nested_tensor_from_tensor_list(position_ids_list, ragged_idx=2)",
            (ROOT / "verl" / "verl" / "workers" / "utils" / "padding.py").read_text(encoding="utf-8"),
        )
        self.assertIn("distillation.distillation_loss.loss_mode=", cli)
        self.assertIn("trainer.n_gpus_per_node={opcd['actor_gpus_per_node']}", cli)
        self.assertIn('training_root / ("pilot" if pilot else "opcd")', cli)
        self.assertIn("trainer.resume_mode={'disable' if pilot else 'auto'}", cli)
        self.assertIn("trainer.save_freq={-1 if pilot else opcd['save_freq']}", cli)
        self.assertEqual(cli.count("trainer.save_freq="), 1)
        self.assertIn("run_training(config, args.data, pilot=args.pilot)", cli)
        self.assertIn("Path(training_root) / args.group", cli)
        self.assertIn("validate_model_pair(", main_ppo)
        self.assertIn("n_gpus_per_node: 2", config)
        self.assertIn("actor_gpus_per_node: 1", config)
        self.assertIn("teacher_gpus_per_node: 1", config)
        self.assertIn("experience_pool", cli)
        self.assertIn('"path", "experience_pool/experience_pool.json"', cli)
        self.assertIn(
            'config.setdefault("experience_extraction", {})["output_path"] = group[',
            cli,
        )
        self.assertIn('"extraction_path"', cli)
        import yaml

        parsed_config = yaml.safe_load(config)
        for group in parsed_config["experience_groups"].values():
            self.assertEqual(
                group["extraction_path"],
                parsed_config["experience_extraction"]["output_path"],
            )
        self.assertNotIn("reference_evaluation:", config)
        self.assertNotIn("experience_selection:", config)
        self.assertIn("device_map: cuda:0", config)
        self.assertNotIn("max_prompt_length: 4096", config)
        self.assertNotIn("experience_max_length:", config)
        self.assertNotIn("pace_experience_max_length", adapter)
        self.assertIn("pace_teacher_max_model_len", adapter)
        evaluation_source = (TRAINER / "pace_evaluation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('{"none", "full_pool"}', evaluation_source)
        self.assertNotIn("resolve_selected_experience_path(config)", evaluation_source)
        self.assertIn(
            '"image_sha256": _content_fingerprint(sample.image_path)',
            evaluation_source,
        )


if __name__ == "__main__":
    unittest.main()
