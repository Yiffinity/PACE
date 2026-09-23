"""Strict per-sample teacher context for the selected-experience OPD runs."""

from collections.abc import Mapping
from copy import deepcopy

from verl.trainer.pace_plus_prompts import BLIND_REASONING_SYSTEM_PROMPT, experience_conditioned_messages


GOLD_TEACHER_INSTRUCTION = """Training-only teacher guidance:
The verified dataset label for this sample is supplied below because the prior blind classification did not pass verification. Use it as the target judgment while assessing the student's continuation. The label is supervision, not visual or textual evidence. Ground the analysis in the attached image and post text; do not invent facts or claim that the label itself proves an interpretation. Ignore any reference that conflicts with the observable evidence. Do not mention this training guidance or the supplied label in the analysis.
<gold_label>{gold_label}</gold_label>"""


def selected_teacher_messages(raw_prompt, context, *, expected_group):
    """Add only chosen references and, for failed rows, gold to the teacher prefix."""
    if not isinstance(context, Mapping):
        raise ValueError("selected OPD requires per-sample teacher_context")
    if not expected_group or context.get("teacher_group") != expected_group:
        raise ValueError("teacher_context belongs to a different teacher group")
    if not isinstance(context.get("sample_id"), str) or not context["sample_id"]:
        raise ValueError("teacher_context requires sample_id")
    verified = context.get("blind_verified")
    gold = context.get("gold_label")
    if type(verified) is not bool:
        raise ValueError("blind_verified must be a boolean")
    if verified and gold not in (None, ""):
        raise ValueError("blind-verified teacher context must not contain gold")
    if not verified and gold not in ("sarcastic", "non-sarcastic"):
        raise ValueError("failed teacher context requires a valid gold label")
    experience = context.get("experience_text")
    if not isinstance(experience, str):
        raise ValueError("teacher_context requires explicit experience_text, including empty selections")
    messages = deepcopy(list(raw_prompt))
    if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
        raise ValueError("selected OPD requires a fresh system/user prompt without history")
    system = messages[0].get("content")
    # VeRL converts even the system string into a typed text block for image rows.
    # Normalize that representation before appending the same offline selection prompt.
    if isinstance(system, list) and len(system) == 1 and system[0].get("type") == "text":
        system = system[0].get("text")
    if system != BLIND_REASONING_SYSTEM_PROMPT:
        raise ValueError("student and teacher must share BLIND_REASONING_SYSTEM_PROMPT")
    messages[0]["content"] = system
    messages = experience_conditioned_messages(messages, experience)
    if not verified:
        messages[0]["content"] += "\n\n" + GOLD_TEACHER_INSTRUCTION.format(gold_label=gold)
    return messages
