from __future__ import annotations

import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .candidates import CandidateValidationError, validate_and_freeze_candidates
from .config import AppConfig, save_config
from .discovery import compute_residual, make_residual_record, rank_residuals
from .encoding import TextEncoder
from .proposer import CandidateProposer
from .scoring import (
    Descriptor,
    EvidenceScorer,
    LikelihoodError,
    classify_or_defer,
    taxonomy_evidence,
)
from .selection import SequentialBayesianSelector
from .taxonomy import Taxonomy
from .types import (
    CycleRecord,
    FrozenCandidate,
    PredictionRecord,
    ResidualRecord,
    StreamItem,
)
from .utils import atomic_write_json, sha256_hex, write_jsonl


@dataclass
class _ActiveCycle:
    null_taxonomy: Taxonomy
    candidates: list[FrozenCandidate]
    selector: SequentialBayesianSelector
    record: CycleRecord


class SBLSEngine:
    def __init__(
        self,
        *,
        config: AppConfig,
        taxonomy: Taxonomy,
        scorer: EvidenceScorer,
        encoder: TextEncoder,
        proposer: CandidateProposer,
    ) -> None:
        self.config = config
        self.taxonomy = taxonomy
        self.scorer = scorer
        self.encoder = encoder
        self.proposer = proposer
        self.residual_buffer: list[ResidualRecord] = []
        self.active_cycle: _ActiveCycle | None = None
        self.cycle_index = 0
        self.predictions: list[dict[str, Any]] = []
        self.cycles: list[dict[str, Any]] = []
        self.proposal_attempts: list[dict[str, Any]] = []

    def _append_prediction(self, record: PredictionRecord) -> None:
        self.predictions.append(asdict(record))

    def _classify_with_scores(
        self,
        *,
        taxonomy: Taxonomy,
        text: str,
        all_taxonomies: list[Taxonomy] | None = None,
    ) -> tuple[Any, dict[int, Any]]:
        taxonomies = all_taxonomies or [taxonomy]
        signature_to_id: dict[tuple[str, str], str] = {}
        descriptors: list[Descriptor] = []
        mappings: dict[int, dict[str, str]] = {}
        for candidate_taxonomy in taxonomies:
            mapping: dict[str, str] = {}
            for node in candidate_taxonomy.labels:
                signature = (node.name, node.description)
                descriptor_id = signature_to_id.get(signature)
                if descriptor_id is None:
                    descriptor_id = f"desc_{sha256_hex(signature)[:24]}"
                    signature_to_id[signature] = descriptor_id
                    descriptors.append(Descriptor(descriptor_id, node.name, node.description))
                mapping[node.id] = descriptor_id
            mappings[id(candidate_taxonomy)] = mapping
        background_id = "__background__"
        descriptors.append(
            Descriptor(
                background_id,
                "background",
                self.config.sbls.background_description,
            )
        )
        scores = self.scorer.score_descriptors(text, descriptors)
        evidences: dict[int, Any] = {}
        for candidate_taxonomy in taxonomies:
            remapped = {
                label_id: scores[descriptor_id]
                for label_id, descriptor_id in mappings[id(candidate_taxonomy)].items()
            }
            evidences[id(candidate_taxonomy)] = taxonomy_evidence(candidate_taxonomy, remapped)
        null_evidence = evidences[id(taxonomy)]
        background = scores[background_id]
        if background.scored_tokens != null_evidence.scored_tokens:
            raise LikelihoodError("Background score token count differs from taxonomy score token count")
        deferral_score = (
            null_evidence.log_mixture - background.log_likelihood
        ) / null_evidence.scored_tokens
        best_label = min(
            null_evidence.label_posteriors,
            key=lambda label_id: (-null_evidence.label_posteriors[label_id], label_id),
        )
        decision = {
            "action": "predict" if deferral_score >= 0.0 else "defer",
            "predicted_label_id": best_label if deferral_score >= 0.0 else None,
            "score": deferral_score,
            "taxonomy_evidence": null_evidence,
            "background_log_likelihood": background.log_likelihood,
        }
        return decision, evidences

    def process(self, item: StreamItem) -> None:
        if self.active_cycle is None:
            self._process_discovery(item)
        else:
            self._process_validation(item)

    def _process_discovery(self, item: StreamItem) -> None:
        decision = classify_or_defer(
            self.taxonomy,
            self.scorer,
            item.text,
            self.config.sbls.background_description,
        )
        self._append_prediction(
            PredictionRecord(
                sample_id=item.id,
                index=item.index,
                timestamp=item.timestamp,
                predicted_label_id=decision.predicted_label_id,
                action=decision.action,
                deferral_score=decision.score,
                taxonomy_version=self.taxonomy.version,
                cycle_index=None,
                phase="discovery",
                metadata=item.metadata,
            )
        )
        if decision.action == "defer":
            residual = compute_residual(
                text=item.text,
                taxonomy=self.taxonomy,
                evidence=decision.taxonomy_evidence,
                encoder=self.encoder,
            )
            record = make_residual_record(
                sample_id=item.id,
                text=item.text,
                arrival_index=item.index,
                token_ids=self.scorer.token_ids(item.text),
                computation=residual,
            )
            if record is not None:
                self.residual_buffer.append(record)
        if len(self.residual_buffer) == self.config.sbls.residual_batch_size:
            self._open_proposal(item.index)
        elif len(self.residual_buffer) > self.config.sbls.residual_batch_size:
            raise RuntimeError("Residual buffer exceeded the configured batch size")

    def _open_proposal(self, proposal_time: int) -> None:
        ranked = rank_residuals(self.residual_buffer)
        if self.config.sbls.exclusive_large_model_residency:
            self.scorer.unload()
        raw_output = ""
        proposal_error: str | None = None
        try:
            raw_output = self.proposer.propose(
                self.taxonomy,
                ranked,
                self.config.sbls.proposal_budget - 1,
            )
            candidates, audit = validate_and_freeze_candidates(
                raw_output=raw_output,
                taxonomy=self.taxonomy,
                ranked_batch=ranked,
                proposal_budget=self.config.sbls.proposal_budget,
                token_counter=self.proposer.token_count,
                max_name_tokens=self.config.proposer.max_name_tokens,
                max_description_tokens=self.config.proposer.max_description_tokens,
            )
        except (CandidateValidationError, RuntimeError, ValueError) as exc:
            candidates = []
            audit = []
            proposal_error = str(exc)
        finally:
            self.proposer.unload()
        attempt = {
            "proposal_time": proposal_time,
            "taxonomy_version": self.taxonomy.version,
            "proposal_sample_ids": [item.record.sample_id for item in ranked],
            "spectral_scores": [item.spectral_score for item in ranked],
            "system_prompt": getattr(self.proposer, "last_system_prompt", None),
            "user_prompt": getattr(self.proposer, "last_user_prompt", None),
            "rendered_prompt": getattr(self.proposer, "last_rendered_prompt", None),
            "raw_output": raw_output,
            "validation_audit": audit,
            "error": proposal_error,
            "opened_cycle": bool(candidates),
        }
        self.proposal_attempts.append(attempt)
        self.residual_buffer = []
        if not candidates:
            return
        self.cycle_index += 1
        selector = SequentialBayesianSelector(
            candidates=candidates,
            alpha=self.config.sbls.alpha,
            cycle_index=self.cycle_index,
            horizon=self.config.sbls.validation_horizon,
        )
        cycle_record = CycleRecord(
            cycle_index=self.cycle_index,
            proposal_time=proposal_time,
            taxonomy_version_before=self.taxonomy.version,
            candidate_ids=[candidate.identifier for candidate in candidates],
            eta=selector.eta,
            boundary=selector.boundary,
            horizon=self.config.sbls.validation_horizon,
            proposal_sample_ids=[item.record.sample_id for item in ranked],
            proposal_payloads=[candidate.canonical_payload for candidate in candidates],
        )
        self.active_cycle = _ActiveCycle(
            null_taxonomy=self.taxonomy.copy(),
            candidates=candidates,
            selector=selector,
            record=cycle_record,
        )

    def _process_validation(self, item: StreamItem) -> None:
        active = self.active_cycle
        if active is None:
            raise RuntimeError("No active cycle")
        all_taxonomies = [active.null_taxonomy, *[candidate.taxonomy for candidate in active.candidates]]
        try:
            decision, evidences = self._classify_with_scores(
                taxonomy=active.null_taxonomy,
                text=item.text,
                all_taxonomies=all_taxonomies,
            )
            null_ll = evidences[id(active.null_taxonomy)].log_mixture
            candidate_ll = {
                candidate.identifier: evidences[id(candidate.taxonomy)].log_mixture
                for candidate in active.candidates
            }
            step = active.selector.update(
                null_log_likelihood=null_ll,
                candidate_log_likelihoods=candidate_ll,
            )
            self._append_prediction(
                PredictionRecord(
                    sample_id=item.id,
                    index=item.index,
                    timestamp=item.timestamp,
                    predicted_label_id=decision["predicted_label_id"],
                    action=decision["action"],
                    deferral_score=decision["score"],
                    taxonomy_version=active.null_taxonomy.version,
                    cycle_index=active.record.cycle_index,
                    phase="validation",
                    metadata=item.metadata,
                )
            )
        except (LikelihoodError, FloatingPointError, OverflowError, ValueError) as exc:
            self._append_prediction(
                PredictionRecord(
                    sample_id=item.id,
                    index=item.index,
                    timestamp=item.timestamp,
                    predicted_label_id=None,
                    action="defer",
                    deferral_score=None,
                    taxonomy_version=active.null_taxonomy.version,
                    cycle_index=active.record.cycle_index,
                    phase="validation",
                    metadata={**item.metadata, "numerical_error": str(exc)},
                )
            )
            self._close_cycle_stay(
                active,
                stopping_time=active.selector.n,
                close_reason=f"invalid_likelihood:{exc}",
            )
            if self.config.runtime.fail_on_invalid_likelihood:
                raise
            return
        active.record.evidence_trajectory.append(
            {
                "stream_index": item.index,
                "n": step.n,
                "log_e_value": step.log_e_value,
                "boundary": step.boundary,
                "log_likelihood_ratios": step.log_likelihood_ratios,
                "crossed": step.crossed,
                "selected_candidate_id": step.selected_candidate_id,
            }
        )
        if step.crossed:
            if step.selected_candidate_id is None:
                raise RuntimeError("A crossing must identify a selected candidate")
            winner = next(
                candidate
                for candidate in active.candidates
                if candidate.identifier == step.selected_candidate_id
            )
            self.taxonomy = winner.taxonomy.copy()
            active.record.decision = winner.spec.operation
            active.record.selected_candidate_id = winner.identifier
            active.record.stopping_time = step.n
            active.record.close_reason = "boundary_crossing"
            active.record.taxonomy_version_after = self.taxonomy.version
            self.cycles.append(asdict(active.record))
            self.active_cycle = None
        elif step.n >= self.config.sbls.validation_horizon:
            self._close_cycle_stay(
                active,
                stopping_time=step.n,
                close_reason="validation_horizon_exhausted",
            )

    def _close_cycle_stay(self, active: _ActiveCycle, *, stopping_time: int, close_reason: str) -> None:
        self.taxonomy = active.null_taxonomy.copy()
        active.record.decision = "stay"
        active.record.selected_candidate_id = None
        active.record.stopping_time = stopping_time
        active.record.close_reason = close_reason
        active.record.taxonomy_version_after = self.taxonomy.version
        self.cycles.append(asdict(active.record))
        self.active_cycle = None

    def finalize(self) -> None:
        if self.active_cycle is not None:
            self._close_cycle_stay(
                self.active_cycle,
                stopping_time=self.active_cycle.selector.n,
                close_reason="stream_ended",
            )

    def run(self, stream: Iterable[StreamItem]) -> Taxonomy:
        for item in stream:
            self.process(item)
        self.finalize()
        return self.taxonomy

    def write_outputs(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_config(self.config, output / "config.resolved.yaml")
        write_jsonl(output / "predictions.jsonl", self.predictions)
        write_jsonl(output / "cycles.jsonl", self.cycles)
        write_jsonl(output / "proposal_attempts.jsonl", self.proposal_attempts)
        atomic_write_json(output / "final_taxonomy.json", self.taxonomy.to_dict())
        metadata = {
            "python": sys.version,
            "platform": platform.platform(),
            "evidence_model": self.config.evidence.model_id,
            "evidence_revision_requested": self.config.evidence.revision,
            "evidence_revision_resolved": getattr(
                self.scorer, "resolved_model_revision", self.config.evidence.revision
            ),
            "tokenizer_revision_requested": self.config.evidence.tokenizer_revision,
            "tokenizer_revision_resolved": getattr(
                self.scorer, "resolved_tokenizer_revision", self.config.evidence.tokenizer_revision
            ),
            "encoder_model": self.config.encoder.model_id,
            "encoder_revision": self.config.encoder.revision,
            "proposer_model": self.config.proposer.model_id,
            "proposer_revision_requested": self.config.proposer.revision,
            "proposer_revision_resolved": getattr(
                self.proposer, "resolved_model_revision", None
            ) or self.config.proposer.revision,
            "proposer_tokenizer_revision_requested": self.config.proposer.tokenizer_revision,
            "proposer_tokenizer_revision_resolved": getattr(
                self.proposer, "resolved_tokenizer_revision", None
            ) or self.config.proposer.tokenizer_revision,
            "prediction_count": len(self.predictions),
            "cycle_count": len(self.cycles),
            "proposal_attempt_count": len(self.proposal_attempts),
            "final_taxonomy_version": self.taxonomy.version,
            "final_taxonomy_hash": sha256_hex(self.taxonomy.to_dict()),
        }
        try:
            import torch

            metadata["torch"] = torch.__version__
            metadata["cuda"] = torch.version.cuda
            metadata["gpu"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            )
        except ImportError:
            metadata["torch"] = None
            metadata["cuda"] = None
            metadata["gpu"] = None
        try:
            import transformers

            metadata["transformers"] = transformers.__version__
        except ImportError:
            metadata["transformers"] = None
        atomic_write_json(output / "run_metadata.json", metadata)
