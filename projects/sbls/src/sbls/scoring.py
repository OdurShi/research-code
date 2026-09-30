from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from .cache import SQLiteCache
from .config import EvidenceModelConfig
from .taxonomy import Taxonomy
from .types import DeferralDecision, SequenceScore, TaxonomyEvidence
from .utils import logsumexp, sha256_hex, stable_text_hash


LABEL_TEMPLATE_VERSION = "sbls-label-template-v1"
LABEL_PREFIX = "Category: {name}\nDefinition: {description}\nText:"


@dataclass(frozen=True)
class Descriptor:
    id: str
    name: str
    description: str


class EvidenceScorer(Protocol):
    def score_descriptors(self, text: str, descriptors: Iterable[Descriptor]) -> dict[str, SequenceScore]: ...

    def token_ids(self, text: str) -> list[int]: ...

    def unload(self) -> None: ...


class LikelihoodError(RuntimeError):
    """Raised when a common-support predictive likelihood is invalid."""


def taxonomy_evidence(
    taxonomy: Taxonomy,
    scores: dict[str, SequenceScore],
) -> TaxonomyEvidence:
    missing = set(taxonomy.label_ids) - set(scores)
    if missing:
        raise LikelihoodError(f"Missing likelihood scores for labels: {sorted(missing)}")
    log_terms: list[float] = []
    label_ll: dict[str, float] = {}
    scored_counts: set[int] = set()
    for node in taxonomy.labels:
        score = scores[node.id]
        if not math.isfinite(score.log_likelihood):
            raise LikelihoodError(f"Non-finite likelihood for label {node.id}")
        if score.scored_tokens <= 0:
            raise LikelihoodError(f"Non-positive scored-token count for label {node.id}")
        label_ll[node.id] = score.log_likelihood
        scored_counts.add(score.scored_tokens)
        log_terms.append(math.log(node.prior) + score.log_likelihood)
    if len(scored_counts) != 1:
        raise LikelihoodError("All descriptions must score the same continuation token sequence")
    log_mix = logsumexp(log_terms)
    posteriors: dict[str, float] = {}
    for node in taxonomy.labels:
        posteriors[node.id] = math.exp(math.log(node.prior) + label_ll[node.id] - log_mix)
    posterior_sum = sum(posteriors.values())
    if not math.isfinite(posterior_sum) or posterior_sum <= 0:
        raise LikelihoodError("Invalid posterior normalization")
    posteriors = {k: v / posterior_sum for k, v in posteriors.items()}
    return TaxonomyEvidence(
        log_mixture=log_mix,
        label_posteriors=posteriors,
        label_log_likelihoods=label_ll,
        scored_tokens=scored_counts.pop(),
    )


def classify_or_defer(
    taxonomy: Taxonomy,
    scorer: EvidenceScorer,
    text: str,
    background_description: str,
) -> DeferralDecision:
    descriptors = [Descriptor(node.id, node.name, node.description) for node in taxonomy.labels]
    background_id = "__background__"
    descriptors.append(Descriptor(background_id, "background", background_description))
    scores = scorer.score_descriptors(text, descriptors)
    evidence = taxonomy_evidence(taxonomy, scores)
    background = scores[background_id]
    if background.scored_tokens != evidence.scored_tokens:
        raise LikelihoodError("Background and taxonomy scores use different token counts")
    score = (evidence.log_mixture - background.log_likelihood) / evidence.scored_tokens
    best_label = min(
        evidence.label_posteriors,
        key=lambda label_id: (-evidence.label_posteriors[label_id], label_id),
    )
    if score >= 0.0:
        return DeferralDecision(
            action="predict",
            predicted_label_id=best_label,
            score=score,
            taxonomy_evidence=evidence,
            background_log_likelihood=background.log_likelihood,
        )
    return DeferralDecision(
        action="defer",
        predicted_label_id=None,
        score=score,
        taxonomy_evidence=evidence,
        background_log_likelihood=background.log_likelihood,
    )


def score_taxonomy_union(
    scorer: EvidenceScorer,
    text: str,
    taxonomies: Iterable[Taxonomy],
) -> tuple[dict[str, SequenceScore], dict[int, TaxonomyEvidence]]:
    taxonomy_list = list(taxonomies)
    descriptor_by_id: dict[str, Descriptor] = {}
    signature_to_score_id: dict[tuple[str, str], str] = {}
    taxonomy_score_ids: dict[int, dict[str, str]] = {}
    for taxonomy in taxonomy_list:
        mapping: dict[str, str] = {}
        for node in taxonomy.labels:
            signature = (node.name, node.description)
            score_id = signature_to_score_id.get(signature)
            if score_id is None:
                score_id = f"desc_{sha256_hex(signature)[:24]}"
                signature_to_score_id[signature] = score_id
                descriptor_by_id[score_id] = Descriptor(score_id, node.name, node.description)
            mapping[node.id] = score_id
        taxonomy_score_ids[id(taxonomy)] = mapping
    union_scores = scorer.score_descriptors(text, descriptor_by_id.values())
    result: dict[int, TaxonomyEvidence] = {}
    for taxonomy in taxonomy_list:
        remapped = {
            label_id: union_scores[score_id]
            for label_id, score_id in taxonomy_score_ids[id(taxonomy)].items()
        }
        result[id(taxonomy)] = taxonomy_evidence(taxonomy, remapped)
    return union_scores, result


class HFCausalEvidenceScorer:
    """Frozen causal-LM continuation likelihood used by SBLS.

    The implementation follows the paper's exact conditioning text and scores only
    input-text plus termination tokens. Prompt tokens are excluded.
    """

    def __init__(self, config: EvidenceModelConfig, *, max_input_tokens: int | None = None) -> None:
        self.config = config
        self.max_input_tokens = int(max_input_tokens or config.max_input_tokens)
        self._tokenizer = None
        self._model = None
        self._resolved_model_revision: str | None = None
        self._resolved_tokenizer_revision: str | None = None
        self._cache = SQLiteCache(config.cache_path) if config.use_cache else None

    @property
    def resolved_model_revision(self) -> str:
        return self._resolved_model_revision or self.config.revision

    @property
    def resolved_tokenizer_revision(self) -> str:
        return self._resolved_tokenizer_revision or self.config.tokenizer_revision

    @property
    def tokenizer(self):
        self._ensure_tokenizer()
        return self._tokenizer

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for HFCausalEvidenceScorer. Install the project dependencies."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.tokenizer_revision,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
            local_files_only=self.config.local_files_only,
        )
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"Tokenizer {self.config.model_id} has no EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._resolved_tokenizer_revision = (
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            or self.config.tokenizer_revision
        )
        self._tokenizer = tokenizer

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._ensure_tokenizer()
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for HFCausalEvidenceScorer"
            ) from exc
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        kwargs = dict(
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=dtype_map[self.config.dtype],
            local_files_only=self.config.local_files_only,
        )
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
        model.eval()
        model.requires_grad_(False)
        model.to(self.config.device)
        self._resolved_model_revision = getattr(model.config, "_commit_hash", None) or self.config.revision
        self._model = model

    def token_ids(self, text: str) -> list[int]:
        self._ensure_tokenizer()
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _termination_ids(self, truncated: bool) -> list[int]:
        self._ensure_tokenizer()
        if not truncated:
            return [int(self._tokenizer.eos_token_id)]
        if self.config.truncation_token_id is not None:
            return [int(self.config.truncation_token_id)]
        ids = list(
            self._tokenizer(self.config.truncation_text, add_special_tokens=False)["input_ids"]
        )
        if not ids:
            raise RuntimeError("Configured truncation_text tokenizes to an empty sequence")
        return [int(x) for x in ids]

    def _encode_conditioned_sequence(self, descriptor: Descriptor, text: str) -> tuple[list[int], list[bool], str]:
        self._ensure_tokenizer()
        prefix = LABEL_PREFIX.format(name=descriptor.name, description=descriptor.description)
        continuation = " " + text
        full = prefix + continuation
        tokenizer = self._tokenizer
        score_start: int
        full_ids: list[int]
        continuation_positions: list[int]
        if getattr(tokenizer, "is_fast", False):
            encoded = tokenizer(
                full,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            full_ids = [int(x) for x in encoded["input_ids"]]
            offsets = encoded["offset_mapping"]
            score_start = len(prefix)
            continuation_positions = [
                index for index, (begin, end) in enumerate(offsets) if int(end) > score_start
            ]
        else:
            full_ids = [int(x) for x in tokenizer(full, add_special_tokens=False)["input_ids"]]
            prefix_ids = [int(x) for x in tokenizer(prefix, add_special_tokens=False)["input_ids"]]
            common = 0
            for left, right in zip(full_ids, prefix_ids):
                if left != right:
                    break
                common += 1
            continuation_positions = list(range(common, len(full_ids)))
        if not continuation_positions:
            # Empty text is still a valid finite sequence; only termination is scored.
            continuation_positions = []
            prompt_end = len(full_ids)
        else:
            prompt_end = continuation_positions[0]
        continuation_ids = [full_ids[i] for i in continuation_positions]
        eos_ids = self._termination_ids(False)
        trunc_ids = self._termination_ids(True)
        if len(eos_ids) >= self.max_input_tokens:
            raise RuntimeError("EOS termination consumes the complete scored-token budget")
        if len(continuation_ids) + len(eos_ids) <= self.max_input_tokens:
            scored_ids = continuation_ids + eos_ids
            terminated_by = "eos"
        else:
            keep = self.max_input_tokens - len(trunc_ids)
            if keep < 0:
                raise RuntimeError("Truncation token sequence exceeds max_input_tokens")
            scored_ids = continuation_ids[:keep] + trunc_ids
            terminated_by = "trunc"
        input_ids = full_ids[:prompt_end] + scored_ids
        score_mask = [False] * prompt_end + [True] * len(scored_ids)
        if not any(score_mask):
            raise RuntimeError("No continuation or termination tokens were selected for scoring")
        if len(input_ids) < 2:
            # Causal scoring requires a context token. Use BOS when the tokenizer supplies one.
            bos = tokenizer.bos_token_id
            if bos is None:
                raise RuntimeError("Cannot score a sequence without a context token")
            input_ids = [int(bos)] + input_ids
            score_mask = [False] + score_mask
        model_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_limit, int) and model_limit < 10**9 and len(input_ids) > model_limit:
            raise RuntimeError(
                f"Conditioned sequence has {len(input_ids)} tokens, exceeding model_max_length={model_limit}"
            )
        return input_ids, score_mask, terminated_by

    def _cache_key(self, descriptor: Descriptor, text: str) -> str:
        return sha256_hex(
            {
                "kind": "causal-likelihood",
                "template": LABEL_TEMPLATE_VERSION,
                "model": self.config.model_id,
                "revision": self.config.revision,
                "tokenizer_revision": self.config.tokenizer_revision,
                "max_input_tokens": self.max_input_tokens,
                "truncation_token_id": self.config.truncation_token_id,
                "truncation_text": self.config.truncation_text,
                "descriptor_name": descriptor.name,
                "descriptor_description": descriptor.description,
                "text_hash": stable_text_hash(text),
            }
        )

    def score_descriptors(self, text: str, descriptors: Iterable[Descriptor]) -> dict[str, SequenceScore]:
        descriptor_list = list(descriptors)
        if not descriptor_list:
            return {}
        if len({d.id for d in descriptor_list}) != len(descriptor_list):
            raise ValueError("Descriptor ids must be unique within one scoring call")
        result: dict[str, SequenceScore] = {}
        missing: list[Descriptor] = []
        cache_keys: dict[str, str] = {}
        for descriptor in descriptor_list:
            cache_key = self._cache_key(descriptor, text)
            cache_keys[descriptor.id] = cache_key
            cached = self._cache.get_score(cache_key) if self._cache is not None else None
            if cached is None:
                missing.append(descriptor)
            else:
                result[descriptor.id] = cached
        if missing:
            computed = self._score_uncached(text, missing)
            result.update(computed)
            if self._cache is not None:
                for descriptor in missing:
                    score = computed[descriptor.id]
                    self._cache.put_score(
                        cache_keys[descriptor.id],
                        score,
                        {
                            "model_id": self.config.model_id,
                            "revision": self._resolved_model_revision or self.config.revision,
                            "descriptor_id": descriptor.id,
                            "text_hash": stable_text_hash(text),
                        },
                    )
        return {descriptor.id: result[descriptor.id] for descriptor in descriptor_list}

    def _score_uncached(self, text: str, descriptors: list[Descriptor]) -> dict[str, SequenceScore]:
        self._ensure_model()
        import torch
        import torch.nn.functional as F

        prepared = [self._encode_conditioned_sequence(d, text) for d in descriptors]
        results: dict[str, SequenceScore] = {}
        for start in range(0, len(descriptors), self.config.batch_size):
            batch_desc = descriptors[start : start + self.config.batch_size]
            batch_prepared = prepared[start : start + self.config.batch_size]
            max_len = max(len(ids) for ids, _, _ in batch_prepared)
            pad_id = int(self._tokenizer.pad_token_id)
            input_tensor = torch.full(
                (len(batch_desc), max_len),
                fill_value=pad_id,
                dtype=torch.long,
                device=self.config.device,
            )
            attention = torch.zeros_like(input_tensor)
            score_mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            terminated_by: list[str] = []
            for row, (ids, mask, termination) in enumerate(batch_prepared):
                length = len(ids)
                input_tensor[row, :length] = torch.tensor(ids, dtype=torch.long, device=self.config.device)
                attention[row, :length] = 1
                score_mask[row, :length] = torch.tensor(mask, dtype=torch.bool, device=self.config.device)
                terminated_by.append(termination)
            with torch.inference_mode():
                outputs = self._model(input_ids=input_tensor, attention_mask=attention, use_cache=False)
                logits = outputs.logits[:, :-1, :]
                targets = input_tensor[:, 1:]
                target_mask = score_mask[:, 1:] & attention[:, 1:].bool()
                selected_logits = logits[target_mask]
                selected_targets = targets[target_mask]
                if selected_targets.numel() == 0:
                    raise LikelihoodError("No target tokens selected in a likelihood batch")
                selected_log_probs = F.log_softmax(selected_logits.float(), dim=-1).gather(
                    1, selected_targets.unsqueeze(1)
                ).squeeze(1)
                cursor = 0
                for row, descriptor in enumerate(batch_desc):
                    count = int(target_mask[row].sum().item())
                    values = selected_log_probs[cursor : cursor + count]
                    cursor += count
                    ll = float(values.sum(dtype=torch.float32).item())
                    if not math.isfinite(ll):
                        raise LikelihoodError(f"Non-finite likelihood for descriptor {descriptor.id}")
                    results[descriptor.id] = SequenceScore(
                        log_likelihood=ll,
                        scored_tokens=count,
                        terminated_by=terminated_by[row],
                    )
        return results

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

    def close(self) -> None:
        self.unload()
        if self._cache is not None:
            self._cache.close()
            self._cache = None
