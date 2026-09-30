from __future__ import annotations

import gc
import json
from typing import Protocol

from .config import ProposerConfig
from .taxonomy import Taxonomy
from .types import RankedResidual


SYSTEM_PROMPT = """You are a deterministic taxonomy edit proposer for an online text classifier.
Return exactly one UTF-8 JSON object and no markdown, commentary, or code fences.
The only allowed top-level field is \"candidates\".
Each candidate must be one of these exact schemas:
1. {\"operation\":\"add\",\"parent_id\":\"<root-or-local-id>\",\"source_label_id\":null,\"new_labels\":[{\"name\":\"...\",\"description\":\"...\"}]}
2. {\"operation\":\"split\",\"parent_id\":\"<source-parent-id>\",\"source_label_id\":\"<local-leaf-id>\",\"new_labels\":[{\"name\":\"...\",\"description\":\"...\"},{\"name\":\"...\",\"description\":\"...\"}]}
Do not rename, merge, delete, reparent, or modify any deployed label. Do not use identifiers that are absent from the supplied local taxonomy. New names must be distinct from deployed names. Propose only coherent recurring semantics supported by multiple ranked examples. Return at most the requested number of candidates, in descending plausibility order."""


class CandidateProposer(Protocol):
    def propose(self, taxonomy: Taxonomy, ranked_batch: list[RankedResidual], max_candidates: int) -> str: ...

    def token_count(self, text: str) -> int: ...

    def unload(self) -> None: ...


class HFCandidateProposer:
    def __init__(self, config: ProposerConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.last_rendered_prompt: str | None = None
        self.resolved_model_revision: str | None = None
        self.resolved_tokenizer_revision: str | None = None

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for HFCandidateProposer") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.tokenizer_revision,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
            local_files_only=self.config.local_files_only,
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self.resolved_tokenizer_revision = (
            getattr(self._tokenizer, "init_kwargs", {}).get("_commit_hash")
            or self.config.tokenizer_revision
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._ensure_tokenizer()
        try:
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("torch, transformers, and bitsandbytes (for NF4) are required") from exc
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        kwargs = {
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype_map[self.config.dtype],
            "local_files_only": self.config.local_files_only,
        }
        if self.config.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype_map[self.config.dtype],
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": self.config.device}
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
        if self.config.quantization == "none":
            model.to(self.config.device)
        model.eval()
        model.requires_grad_(False)
        self.resolved_model_revision = getattr(model.config, "_commit_hash", None) or self.config.revision
        self._model = model

    def token_count(self, text: str) -> int:
        self._ensure_tokenizer()
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _truncate_example(self, text: str) -> str:
        self._ensure_tokenizer()
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.config.max_example_tokens:
            return text
        ids = ids[: self.config.max_example_tokens]
        return self._tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _build_user_prompt(
        self,
        taxonomy: Taxonomy,
        ranked_batch: list[RankedResidual],
        max_candidates: int,
    ) -> str:
        local_ids = sorted({label_id for item in ranked_batch for label_id in item.record.local_label_ids})
        local_labels = []
        for label_id in local_ids:
            node = taxonomy.get(label_id)
            local_labels.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "parent_id": taxonomy.root_id if node.parent_id is None else node.parent_id,
                    "is_leaf": taxonomy.is_leaf(node.id),
                }
            )
        examples = [
            {
                "rank": item.rank,
                "spectral_score": item.spectral_score,
                "sample_id": item.record.sample_id,
                "text": self._truncate_example(item.record.text),
                "posterior_local_label_ids": list(item.record.local_label_ids),
            }
            for item in ranked_batch
        ]
        payload = {
            "root_id": taxonomy.root_id,
            "maximum_candidates": max_candidates,
            "local_taxonomy": local_labels,
            "local_edges": taxonomy.local_edges(local_ids),
            "ranked_deferred_examples": examples,
        }
        return (
            "Construct frozen local alternatives from the proposal batch below. "
            "Use only add or binary split and obey every schema constraint.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    def propose(self, taxonomy: Taxonomy, ranked_batch: list[RankedResidual], max_candidates: int) -> str:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self._ensure_model()
        import torch

        user_prompt = self._build_user_prompt(taxonomy, ranked_batch, max_candidates)
        self.last_system_prompt = SYSTEM_PROMPT
        self.last_user_prompt = user_prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"
        self.last_rendered_prompt = rendered
        encoded = self._tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens > self.config.max_prompt_tokens:
            raise RuntimeError(
                f"Proposer prompt contains {prompt_tokens} tokens, exceeding max_prompt_tokens={self.config.max_prompt_tokens}. "
                "All B proposal examples are retained; increase the configured prompt limit or use a longer-context proposer."
            )
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )
        output_ids = generated[0, prompt_tokens:]
        return self._tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                return
