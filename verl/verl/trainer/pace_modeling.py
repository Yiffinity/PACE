"""Local Qwen multimodal inference used by extraction and evaluation."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from verl.trainer.pace_preflight import assert_runtime_multimodal, inspect_checkpoint


class ModelLoadError(RuntimeError):
    pass



def _empty_cuda_cache_safely(torch: Any, device: Any = None) -> None:
    if not torch.cuda.is_available():
        return
    try:
        if device is not None and str(device).startswith("cuda"):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:
        pass

def _model_placements(model: Any) -> set[str]:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        return {
            f"cuda:{device}" if isinstance(device, int) else str(device)
            for device in device_map.values()
        }
    return {str(parameter.device) for parameter in model.parameters()}


def _assert_supported_device_placement(model: Any, metadata: Mapping[str, Any], *, role: str) -> None:
    placements = _model_placements(model) - {"meta"}
    if metadata.get("model_type") == "qwen3_5" and len(placements) > 1:
        devices = ", ".join(sorted(placements))
        raise ModelLoadError(
            f"{role} Qwen3.5 was split across devices ({devices}); this makes the hybrid "
            "Gated DeltaNet generate non-finite logits in local multimodal inference. Set "
            "generation.device_map to one device such as 'cuda:0'."
        )


class MultimodalGenerator:
    def __init__(
        self,
        model_path: str | Path,
        *,
        role: str,
        device_map: str | Mapping[str, Any] = "auto",
        max_new_tokens: int = 768,
        deterministic: bool = True,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        empty_cache_after_generate: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.role = role
        self.max_new_tokens = max_new_tokens
        self.deterministic = deterministic
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.empty_cache_after_generate = bool(empty_cache_after_generate)
        metadata = inspect_checkpoint(self.model_path)
        self.metadata = metadata

        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ModelLoadError("transformers and torch are required for model inference") from exc

        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=False, local_files_only=True
            )
        except Exception as exc:
            raise ModelLoadError(f"cannot load multimodal processor from {self.model_path}: {exc}") from exc

        architecture = metadata["architecture"]
        model_class = getattr(transformers, architecture, None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_class is None:
            raise ModelLoadError(
                f"installed transformers does not provide {architecture} or an image-to-text auto model; "
                "install a Transformers release supporting Qwen3.5"
            )
        try:
            self.model = model_class.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=False,
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelLoadError(f"cannot load {architecture} from {self.model_path}: {exc}") from exc
        self.model.eval()
        self.model.requires_grad_(False)
        _assert_supported_device_placement(self.model, metadata, role=role)
        assert_runtime_multimodal(self.model, self.processor, role=role)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ModelLoadError(f"{role} parameters were not frozen for inference")
        self._torch = torch

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.metadata, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _device_inputs(self, inputs: Any) -> Any:
        device = getattr(self.model, "device", None)
        if device is None:
            return inputs
        return inputs.to(device)

    def _render_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def count_input_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        prompt = self._render_prompt(messages)
        inputs = self.processor(text=[prompt], return_tensors="pt")
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ModelLoadError(f"{self.role} processor did not return input_ids")
        return int(input_ids.shape[-1])


    def generate(self, messages: Sequence[Mapping[str, Any]], *, image_path: str | Path | None) -> str:
        from PIL import Image

        inputs = None
        output_ids = None
        generated = None
        image = None
        try:
            prompt = self._render_prompt(messages)
            if image_path is not None:
                with Image.open(Path(image_path)) as opened:
                    image = opened.convert("RGB").copy()
            kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
            if image is not None:
                kwargs["images"] = [image]
            inputs = self.processor(**kwargs)
            if image_path is not None:
                image_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos"}
                if not image_keys.intersection(inputs.keys()):
                    raise ModelLoadError(f"{self.role} processor dropped image inputs for {image_path}")
            inputs = self._device_inputs(inputs)
            generation = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": not self.deterministic,
            }
            # Some merged checkpoints retain only end-of-text in generation_config,
            # while their chat tokenizer ends assistant turns with a different ID.
            # Honor both so a completed answer cannot continue into fabricated turns.
            stop_ids = []
            for source in (getattr(self.model, "generation_config", None),
                           getattr(self.processor, "tokenizer", None)):
                value = getattr(source, "eos_token_id", None)
                stop_ids.extend(value if isinstance(value, (list, tuple)) else [value])
            stop_ids = list(dict.fromkeys(value for value in stop_ids if isinstance(value, int)))
            if stop_ids:
                generation["eos_token_id"] = stop_ids
            if not self.deterministic:
                generation.update({"temperature": 0.7, "top_p": 0.9})
            with self._torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation)
            input_length = inputs["input_ids"].shape[-1]
            generated = output_ids[:, input_length:]
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        finally:
            if image is not None:
                image.close()
            if self.empty_cache_after_generate:
                del generated, output_ids, inputs
                gc.collect()
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()

    def close(self) -> None:
        model = getattr(self, "model", None)
        processor = getattr(self, "processor", None)
        if model is not None:
            del self.model
        if processor is not None:
            del self.processor
        gc.collect()
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "MultimodalGenerator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class VllmTextGenerator:
    """Single-request text inference with reusable vLLM prefix caches."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        role: str,
        max_new_tokens: int,
        deterministic: bool,
        chat_template_kwargs: Mapping[str, Any] | None,
        max_model_len: int,
        max_num_seqs: int,
        gpu_memory_utilization: float,
        enable_prefix_caching: bool = True,
        prefix_caching_hash_algo: str = "xxhash",
        mamba_cache_mode: str = "align",
        enforce_eager: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.role = role
        self.max_new_tokens = int(max_new_tokens)
        self.deterministic = bool(deterministic)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.json_schema: Mapping[str, Any] | None = None
        self.last_num_cached_tokens = 0
        self.total_num_cached_tokens = 0
        self.metadata = inspect_checkpoint(self.model_path)
        try:
            transformers = importlib.import_module("transformers")
            vllm = importlib.import_module("vllm")
            self._sampling = importlib.import_module("vllm.sampling_params")
        except ImportError as exc:
            raise ModelLoadError("transformers and vllm are required for vLLM text inference") from exc

        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=False, local_files_only=True
        )
        engine_kwargs: dict[str, Any] = {
            "model": str(self.model_path),
            "dtype": "bfloat16",
            "max_model_len": int(max_model_len),
            "max_num_seqs": int(max_num_seqs),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": bool(enforce_eager),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "prefix_caching_hash_algo": str(prefix_caching_hash_algo),
            "mamba_cache_mode": str(mamba_cache_mode),
            "disable_log_stats": True,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
        }
        try:
            self.llm = vllm.LLM(**engine_kwargs)
        except Exception as exc:
            raise ModelLoadError(f"cannot load {role} with vLLM from {self.model_path}: {exc}") from exc

    def _render_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def count_input_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        prompt = self._render_prompt(messages)
        inputs = self.processor(text=[prompt], return_tensors="pt")
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ModelLoadError(f"{self.role} processor did not return input_ids")
        return int(input_ids.shape[-1])

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        image_path: str | Path | None,
    ) -> str:
        if image_path is not None:
            raise ModelLoadError(f"{self.role} accepts text-only requests")
        params: dict[str, Any] = {
            "temperature": 0.0 if self.deterministic else 0.7,
            "top_p": 1.0 if self.deterministic else 0.9,
            "max_tokens": int(self.max_new_tokens),
        }
        if self.json_schema is not None:
            params["structured_outputs"] = self._sampling.StructuredOutputsParams(
                json=dict(self.json_schema)
            )
        sampling_params = self._sampling.SamplingParams(**params)
        outputs = self.llm.generate(
            [{"prompt": self._render_prompt(messages)}],
            sampling_params,
            use_tqdm=False,
        )
        output = outputs[0]
        self.last_num_cached_tokens = int(output.num_cached_tokens or 0)
        self.total_num_cached_tokens += self.last_num_cached_tokens
        return output.outputs[0].text.strip()

    def close(self) -> None:
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()

    def __enter__(self) -> "VllmTextGenerator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class VllmMultimodalGenerator:
    """Batched local multimodal inference with post-thinking JSON constraints."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        role: str,
        max_new_tokens: int,
        deterministic: bool,
        chat_template_kwargs: Mapping[str, Any] | None,
        max_model_len: int,
        max_num_seqs: int,
        gpu_memory_utilization: float,
        thinking_token_budget: int | None,
        json_schema: Mapping[str, Any],
        enforce_eager: bool = True,
        require_thinking: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.role = role
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if require_thinking and self.chat_template_kwargs.get("enable_thinking") is not True:
            raise ModelLoadError(f"{role} requires enable_thinking=true")
        self.metadata = inspect_checkpoint(self.model_path)
        try:
            transformers = importlib.import_module("transformers")
            vllm = importlib.import_module("vllm")
            sampling = importlib.import_module("vllm.sampling_params")
        except ImportError as exc:
            raise ModelLoadError("transformers and vllm are required for batched model inference") from exc

        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=False, local_files_only=True
        )
        engine_kwargs: dict[str, Any] = {
            "model": str(self.model_path),
            "dtype": "bfloat16",
            "max_model_len": int(max_model_len),
            "max_num_seqs": int(max_num_seqs),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": bool(enforce_eager),
            "disable_log_stats": True,
        }
        if thinking_token_budget is not None:
            engine_kwargs["reasoning_parser"] = "qwen3"
        try:
            self.llm = vllm.LLM(**engine_kwargs)
        except Exception as exc:
            raise ModelLoadError(f"cannot load {role} with vLLM from {self.model_path}: {exc}") from exc

        structured = sampling.StructuredOutputsParams(json=dict(json_schema))
        params: dict[str, Any] = {
            "temperature": 0.0 if deterministic else 0.7,
            "top_p": 1.0 if deterministic else 0.9,
            "max_tokens": int(max_new_tokens),
            "thinking_token_budget": thinking_token_budget,
            "structured_outputs": structured,
        }
        self.sampling_params = vllm.SamplingParams(**params)

    def generate_batch(
        self,
        requests: Sequence[tuple[Sequence[Mapping[str, Any]], str | Path | None]],
    ) -> list[str]:
        from PIL import Image

        prompts: list[dict[str, Any]] = []
        images: list[Any] = []
        for messages, image_path in requests:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            request: dict[str, Any] = {"prompt": prompt}
            if image_path is not None:
                resolved = Path(image_path).expanduser().resolve()
                with Image.open(resolved) as opened:
                    image = opened.convert("RGB").copy()
                images.append(image)
                request["multi_modal_data"] = {"image": image}
                request["multi_modal_uuids"] = {"image": str(resolved)}
            prompts.append(request)
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def generate(self, messages: Sequence[Mapping[str, Any]], *, image_path: str | Path | None) -> str:
        return self.generate_batch([(messages, image_path)])[0]

    def close(self) -> None:
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()

    def __enter__(self) -> "VllmMultimodalGenerator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
