from __future__ import annotations

from pathlib import Path

import numpy as np

from sbls.config import AppConfig
from sbls.engine import SBLSEngine
from sbls.scoring import Descriptor
from sbls.taxonomy import Taxonomy
from sbls.types import LabelNode, SequenceScore, StreamItem


class MockScorer:
    def score_descriptors(self, text, descriptors):
        result = {}
        lower = text.casefold()
        for descriptor in descriptors:
            desc = (descriptor.name + " " + descriptor.description).casefold()
            overlap = any(token in desc for token in lower.split())
            if descriptor.id == "__background__":
                ll = -3.0
            elif overlap:
                ll = -0.1
            else:
                ll = -10.0
            result[descriptor.id] = SequenceScore(ll, 2, "eos")
        return result

    def token_ids(self, text):
        return [ord(char) for char in text]

    def unload(self):
        return None


class MockEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.casefold()
            if "refund" in lower:
                vector = np.array([1.0, 0.0, 0.0])
            elif "login" in lower:
                vector = np.array([0.0, 1.0, 0.0])
            else:
                vector = np.array([0.0, 0.0, 1.0])
            vectors.append(vector)
        return np.stack(vectors)

    def unload(self):
        return None


class MockProposer:
    def propose(self, taxonomy, ranked_batch, max_candidates):
        return '{"candidates":[{"operation":"add","parent_id":"__root__","source_label_id":null,"new_labels":[{"name":"card dispute","description":"unauthorized card dispute charge"}]}]}'

    def token_count(self, text):
        return len(text.split())

    def unload(self):
        return None


def test_end_to_end_commit(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "sbls": {
                "alpha": 0.5,
                "residual_batch_size": 2,
                "proposal_budget": 2,
                "validation_horizon": 4,
                "exclusive_large_model_residency": True,
            },
            "runtime": {"output_dir": str(tmp_path), "fail_on_invalid_likelihood": True},
        }
    )
    taxonomy = Taxonomy(
        [
            LabelNode("refund", "refund", "merchant refund", prior=0.5),
            LabelNode("login", "login", "account login", prior=0.5),
        ]
    )
    engine = SBLSEngine(
        config=config,
        taxonomy=taxonomy,
        scorer=MockScorer(),
        encoder=MockEncoder(),
        proposer=MockProposer(),
    )
    stream = [
        StreamItem("1", "refund", 1),
        StreamItem("2", "card dispute", 2),
        StreamItem("3", "card charge", 3),
        StreamItem("4", "unauthorized card dispute", 4),
    ]
    engine.run(stream)
    assert engine.taxonomy.size == 3
    assert len(engine.cycles) == 1
    assert engine.cycles[0]["decision"] == "add"
    assert len(engine.predictions) == 4
