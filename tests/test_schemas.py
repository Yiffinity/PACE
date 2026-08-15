import pytest
from pydantic import ValidationError

from pace.schemas import (
    ExperienceReflection,
    ExperienceUtility,
    Label,
    SarcasmAnalysis,
    parse_sarcasm_analysis,
)


def test_strict_sarcasm_analysis() -> None:
    parsed = parse_sarcasm_analysis(
        '{"visual_evidence":"A failed result.",'
        '"textual_evidence":"The text says great.",'
        '"explanation":"The positive wording reverses the observed outcome.",'
        '"label":"sarcastic"}'
    )
    assert parsed.label is Label.SARCASTIC
    assert parsed.label.binary == 1


def test_markdown_fence_is_not_accepted_as_json() -> None:
    with pytest.raises(ValidationError):
        parse_sarcasm_analysis(
            '```json\n{"visual_evidence":"x","textual_evidence":"y",'
            '"explanation":"z","label":"sarcastic"}\n```'
        )


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SarcasmAnalysis(
            visual_evidence="x",
            textual_evidence="y",
            explanation="z",
            label="ironic",
        )


def test_reflection_may_have_no_transferable_experience() -> None:
    reflection = ExperienceReflection(
        corrected_analysis=SarcasmAnalysis(
            visual_evidence="x",
            textual_evidence="y",
            explanation="z",
            label=Label.NON_SARCASTIC,
        ),
        has_new_rule=False,
        experience="  ",
    )
    assert reflection.experience is None


def test_utility_boundary_matches_method() -> None:
    assert ExperienceUtility(
        relevant_score=0.1,
        global_score=0.0,
        relevant_deltas=(0.1,),
        global_deltas=(0.0,),
    ).accepted
    assert not ExperienceUtility(
        relevant_score=0.0,
        global_score=1.0,
        relevant_deltas=(0.0,),
        global_deltas=(1.0,),
    ).accepted
