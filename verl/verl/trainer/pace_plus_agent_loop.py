"""PACE+ experience-conditioned teacher scoring for VeRL online distillation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import ray
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopOutput, AgentLoopWorker
from verl.trainer.pace_plus_pool import load_active_experience_text
from verl.trainer.pace_plus_prompts import experience_conditioned_messages
from verl.trainer.pace_plus_selected_context import selected_teacher_messages
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


def align_teacher_response(
    *,
    student_prompt_ids: list[int],
    response_ids: list[int],
    teacher_prompt_length: int,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher scores from a different prompt to the student's sequence."""
    sequence_length = len(student_prompt_ids) + len(response_ids)
    if teacher_ids.ndim != 2 or teacher_logprobs.ndim != 2:
        raise RuntimeError("teacher IDs and logprobs must both have shape [sequence, topk]")
    if teacher_ids.shape != teacher_logprobs.shape:
        raise RuntimeError("teacher IDs and logprobs have different shapes")
    if teacher_ids.shape[1] != 1:
        raise RuntimeError("PACE_PLUS k3 distillation requires sampled-token teacher logprobs (topk=1)")

    response_length = len(response_ids)
    if not student_prompt_ids or teacher_prompt_length < 1 or not response_ids:
        raise RuntimeError("teacher alignment requires nonempty prompts and response")
    source_start = teacher_prompt_length - 1
    source_end = source_start + response_length
    response_teacher_ids = teacher_ids[source_start:source_end]
    response_teacher_logprobs = teacher_logprobs[source_start:source_end]
    if not torch.isfinite(response_teacher_logprobs).all():
        raise RuntimeError("teacher response logprobs must be finite")
    expected_ids = torch.tensor(response_ids, dtype=response_teacher_ids.dtype).unsqueeze(-1)
    if response_teacher_ids.shape != expected_ids.shape or not torch.equal(response_teacher_ids.cpu(), expected_ids):
        raise RuntimeError("teacher must score the exact student-generated response token IDs")

    aligned_ids = torch.full((sequence_length, 1), pad_token_id, dtype=teacher_ids.dtype)
    aligned_logprobs = torch.zeros((sequence_length, 1), dtype=teacher_logprobs.dtype)
    target_start = len(student_prompt_ids) - 1
    target_end = target_start + response_length
    aligned_ids[target_start:target_end] = response_teacher_ids
    aligned_logprobs[target_start:target_end] = response_teacher_logprobs
    return aligned_ids, aligned_logprobs


class PacePlusAgentLoopWorker(AgentLoopWorker):
    """Score student responses with the frozen teacher under consolidated experience."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.distillation_enabled:
            raise RuntimeError("PACE_PLUS requires distillation.enabled=True")
        self.selected_context = bool(OmegaConf.select(self.config, "trainer.pace_plus_selected_context", default=False))
        self.teacher_group = OmegaConf.select(self.config, "trainer.pace_plus_teacher_group")
        if self.selected_context:
            if not self.teacher_group:
                raise RuntimeError("selected OPD requires trainer.pace_plus_teacher_group")
            self.experience_text = ""
            return
        experience_path = OmegaConf.select(self.config, "trainer.pace_plus_experience_path")
        if not experience_path:
            raise RuntimeError("PACE_PLUS requires trainer.pace_plus_experience_path")
        self.experience_text = load_active_experience_text(experience_path)

    def _teacher_prompt_ids(self, output: AgentLoopOutput, raw_prompt: Any, context=None) -> list[int]:
        if self.selected_context:
            messages = selected_teacher_messages(raw_prompt, context, expected_group=self.teacher_group)
        else:
            messages = experience_conditioned_messages(deepcopy(list(raw_prompt)), self.experience_text)
        apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        if self.processor is None:
            tokenized = apply_chat_template(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            )
            return normalize_token_ids(tokenized)

        rendered = apply_chat_template(
            self.processor,
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        multi_modal_data = output.multi_modal_data or {}
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[rendered],
            images=multi_modal_data.get("images"),
            videos=multi_modal_data.get("videos"),
            audio=multi_modal_data.get("audios"),
            mm_processor_kwargs=output.mm_processor_kwargs,
        )
        return normalize_token_ids(model_inputs["input_ids"])

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if validate:
            return
        if not response_ids:
            raise RuntimeError("PACE_PLUS received an empty student response")
        if sample_kwargs is None or "raw_prompt" not in sample_kwargs:
            raise RuntimeError("PACE_PLUS teacher scoring requires the original raw_prompt")

        extra_info = sample_kwargs.get("extra_info", {})
        context = extra_info.get("teacher_context")
        if self.selected_context and (not context or context.get("sample_id") != extra_info.get("sample_id")):
            raise RuntimeError("teacher context is missing or does not match the student sample")
        teacher_prompt_ids = self._teacher_prompt_ids(output, sample_kwargs["raw_prompt"], context)
        teacher_max_length = int(
            OmegaConf.select(self.config, "trainer.pace_plus_teacher_max_model_len", default=0)
        )
        if teacher_max_length and len(teacher_prompt_ids) + len(response_ids) + 1 > teacher_max_length:
            raise RuntimeError(
                "experience-conditioned teacher sequence exceeds "
                f"pace_plus_teacher_max_model_len={teacher_max_length}"
            )

        routing_key = None
        routing_value = sample_kwargs.get(self.teacher_key)
        if routing_value is not None:
            routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
        teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=teacher_prompt_ids + response_ids,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            routing_key=routing_key,
        )
        aligned_ids, aligned_logprobs = align_teacher_response(
            student_prompt_ids=prompt_ids,
            response_ids=response_ids,
            teacher_prompt_length=len(teacher_prompt_ids),
            teacher_ids=teacher_ids,
            teacher_logprobs=teacher_logprobs,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output.extra_fields["teacher_ids"] = aligned_ids
        output.extra_fields["teacher_logprobs"] = aligned_logprobs


class PacePlusAgentLoopManager(AgentLoopManager):
    """Install the PACE+ worker while retaining VeRL's native manager lifecycle."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(PacePlusAgentLoopWorker)
        super().__init__(*args, **kwargs)
