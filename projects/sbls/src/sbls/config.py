from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class EvidenceModelConfig(StrictModel):
    model_id: str = "Qwen/Qwen3-8B"
    revision: str = "main"
    tokenizer_revision: str = "main"
    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    batch_size: int = Field(default=4, ge=1)
    max_input_tokens: int = Field(default=128, ge=2)
    long_input_max_tokens: int = Field(default=512, ge=2)
    long_dataset_names: tuple[str, ...] = (
        "20newsgroups",
        "yahoo_answers",
        "x_topic",
        "dbpedia",
    )
    truncation_token_id: int | None = None
    truncation_text: str = "<trunc>"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    cache_path: str = "cache/sbls.sqlite3"
    use_cache: bool = True
    local_files_only: bool = False


class EncoderConfig(StrictModel):
    model_id: str = "BAAI/bge-m3"
    revision: str = "main"
    device: str = "cuda"
    batch_size: int = Field(default=32, ge=1)
    cache_path: str = "cache/sbls.sqlite3"
    use_cache: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False


class ProposerConfig(StrictModel):
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    revision: str = "main"
    tokenizer_revision: str = "main"
    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    quantization: Literal["none", "nf4"] = "nf4"
    trust_remote_code: bool = False
    local_files_only: bool = False
    max_new_tokens: int = Field(default=512, ge=1)
    max_name_tokens: int = Field(default=8, ge=1)
    max_description_tokens: int = Field(default=48, ge=1)
    max_example_tokens: int = Field(default=256, ge=8)
    max_prompt_tokens: int = Field(default=30000, ge=1024)


class SBLSConfig(StrictModel):
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    residual_batch_size: int = Field(default=64, ge=2)
    proposal_budget: int = Field(default=4, ge=2)
    validation_horizon: int = Field(default=256, ge=1)
    background_description: str = (
        "a text from the deployment domain without assuming any of the listed categories"
    )
    exclusive_large_model_residency: bool = True


class RuntimeConfig(StrictModel):
    seed: int = 13
    dataset_name: str = "generic"
    output_dir: str = "runs/sbls"
    log_every: int = Field(default=25, ge=1)
    fail_on_invalid_likelihood: bool = False


class AppConfig(StrictModel):
    evidence: EvidenceModelConfig = EvidenceModelConfig()
    encoder: EncoderConfig = EncoderConfig()
    proposer: ProposerConfig = ProposerConfig()
    sbls: SBLSConfig = SBLSConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @field_validator("runtime")
    @classmethod
    def normalize_output_dir(cls, value: RuntimeConfig) -> RuntimeConfig:
        return value


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    return AppConfig.model_validate(raw)


def save_config(config: AppConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
