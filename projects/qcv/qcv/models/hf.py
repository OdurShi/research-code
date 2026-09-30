from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..types import QuestionSpec
from .media import load_media


@dataclass(frozen=True)
class ModelLoadOptions:
    model_name_or_path: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attention_implementation: str | None = None
    max_batch_candidates: int = 0


class HuggingFaceVLMScorer:
    """Teacher-forced complete-alphabet scorer for Hugging Face VLMs."""

    def __init__(self, options: ModelLoadOptions):
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFaceVLMScorer requires torch and transformers. Install the project with the 'vlm' extra."
            ) from exc
        self.torch = torch
        self.options = options
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(options.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported dtype {options.dtype!r}")
        self.processor = AutoProcessor.from_pretrained(
            options.model_name_or_path,
            trust_remote_code=options.trust_remote_code,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": options.trust_remote_code,
        }
        if options.attention_implementation:
            load_kwargs["attn_implementation"] = options.attention_implementation
        model = None
        load_errors: list[str] = []
        loader_names = (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        )
        for loader_name in loader_names:
            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(options.model_name_or_path, **load_kwargs)
                break
            except Exception as exc:
                load_errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
        if model is None:
            details = " | ".join(load_errors) if load_errors else "no compatible auto-model class"
            raise RuntimeError(
                f"Could not load {options.model_name_or_path!r} with a supported Hugging Face "
                f"multimodal causal model class. Details: {details}"
            )
        self.model = model.eval().to(options.device)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def _chat_prefix(self, question: QuestionSpec, media_type: str) -> str:
        alphabet = ", ".join(question.role.alphabet)
        instruction = (
            f"{question.text}\n"
            f"Return exactly one canonical answer from this finite set: [{alphabet}].\n"
            "Answer:"
        )
        if hasattr(self.processor, "apply_chat_template"):
            content: list[dict[str, Any]] = []
            if media_type == "image":
                content.append({"type": "image"})
            elif media_type == "video":
                content.append({"type": "video"})
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            try:
                rendered = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered = None
            if rendered is not None:
                return str(rendered)
        visual_token = "<image>\n" if media_type == "image" else "<video>\n"
        return visual_token + instruction

    @staticmethod
    def _find_last_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
        if not subsequence or len(subsequence) > len(sequence):
            return None
        for start in range(len(sequence) - len(subsequence), -1, -1):
            if sequence[start : start + len(subsequence)] == subsequence:
                return start, start + len(subsequence)
        return None

    def _processor_call(self, texts: Sequence[str], media_type: str, media_value: Any) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {
            "text": list(texts),
            "padding": True,
            "return_tensors": "pt",
        }
        if media_type == "image":
            kwargs["images"] = [media_value for _ in texts]
        elif media_type == "video":
            kwargs["videos"] = [media_value for _ in texts]
        try:
            return self.processor(**kwargs)
        except TypeError:
            if media_type == "video":
                kwargs.pop("videos", None)
                kwargs["images"] = [media_value for _ in texts]
            return self.processor(**kwargs)

    def score_question(self, question: QuestionSpec, media: Mapping[str, Any]) -> tuple[float, ...]:
        torch = self.torch
        media_type, media_value = load_media(media)
        prefix = self._chat_prefix(question, media_type)
        answers = list(question.role.alphabet)
        scores: list[float] = []
        batch_size = self.options.max_batch_candidates or len(answers)
        if batch_size < 1:
            raise ValueError("max_batch_candidates must be zero or a positive integer")
        for start in range(0, len(answers), batch_size):
            batch_answers = answers[start : start + batch_size]
            answer_suffixes = [" " + answer for answer in batch_answers]
            texts = [prefix + suffix for suffix in answer_suffixes]
            encoded = self._processor_call(texts, media_type, media_value)
            encoded = {
                key: value.to(self.options.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask")
            if input_ids is None:
                raise RuntimeError("The processor did not return input_ids")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                outputs = self.model(**encoded)
                logits = outputs.logits
                log_probabilities = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            for row, suffix in enumerate(answer_suffixes):
                nonpad_positions = (attention_mask[row] != 0).nonzero(as_tuple=False).flatten()
                valid_ids = input_ids[row, nonpad_positions].tolist()
                suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
                location = self._find_last_subsequence(valid_ids, list(suffix_ids))
                if location is None:
                    raise RuntimeError(
                        f"Could not locate answer tokens for {batch_answers[row]!r} in the model input"
                    )
                token_start, token_end = location
                if token_start == 0:
                    raise RuntimeError("Answer begins at the first token; no causal context is available")
                token_logprobs: list[float] = []
                for valid_position in range(token_start, token_end):
                    token_id = valid_ids[valid_position]
                    padded_position = int(nonpad_positions[valid_position].item())
                    if padded_position == 0:
                        raise RuntimeError("A scored answer token has no causal predecessor")
                    token_logprobs.append(
                        float(log_probabilities[row, padded_position - 1, token_id].item())
                    )
                if not token_logprobs:
                    raise RuntimeError(f"Answer {batch_answers[row]!r} produced no scoreable tokens")
                scores.append(float(np.mean(token_logprobs)))
        return tuple(scores)

    def generate(
        self,
        question_text: str,
        media: Mapping[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> str:
        torch = self.torch
        media_type, media_value = load_media(media)
        question = QuestionSpec(
            question_id="generation",
            text=question_text,
            variable="generation",
            role=self._free_text_role(),
            template_index=0,
            is_target=True,
        )
        prefix = self._chat_prefix(question, media_type).replace(
            "Return exactly one canonical answer from this finite set: [free text].\n", ""
        )
        encoded = self._processor_call([prefix], media_type, media_value)
        encoded = {
            key: value.to(self.options.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        input_length = int(encoded["input_ids"].shape[1])
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "num_beams": num_beams,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", None),
        }
        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        generated = output[0, input_length:]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()

    @staticmethod
    def _free_text_role():
        from ..types import RoleSpec

        return RoleSpec(name="free_text", alphabet=("free text",), answer_type="categorical")
