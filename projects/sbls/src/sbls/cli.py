from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .encoding import BGETextEncoder
from .engine import SBLSEngine
from .io import load_stream, load_taxonomy
from .metrics import evaluate_edit_events, evaluate_prequential_predictions
from .proposer import HFCandidateProposer
from .scoring import HFCausalEvidenceScorer
from .streams import (
    generate_add_stream,
    generate_anomaly_stream,
    generate_covariate_drift_stream,
    generate_prior_drift_stream,
    generate_split_stream,
    load_label_descriptions,
    load_labeled_examples,
)
from .utils import atomic_write_json, read_json, read_jsonl, set_global_seed, write_jsonl


app = typer.Typer(no_args_is_help=True, add_completion=False, help="Sequential Bayesian Label-Space Selection")


@app.command("run")
def run_command(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    taxonomy_path: Path = typer.Option(..., "--taxonomy", exists=True, dir_okay=False),
    stream_path: Path = typer.Option(..., "--stream", exists=True, dir_okay=False),
    output_dir: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    """Run end-to-end Sequential Bayesian Label-Space Selection."""
    config = load_config(config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"output_dir": str(output_dir)})}
        )
    max_tokens = (
        config.evidence.long_input_max_tokens
        if config.runtime.dataset_name in config.evidence.long_dataset_names
        else config.evidence.max_input_tokens
    )
    set_global_seed(config.runtime.seed)
    taxonomy = load_taxonomy(taxonomy_path)
    scorer = HFCausalEvidenceScorer(config.evidence, max_input_tokens=max_tokens)
    encoder = BGETextEncoder(config.encoder)
    proposer = HFCandidateProposer(config.proposer)
    engine = SBLSEngine(
        config=config,
        taxonomy=taxonomy,
        scorer=scorer,
        encoder=encoder,
        proposer=proposer,
    )
    try:
        engine.run(load_stream(stream_path))
        engine.write_outputs(config.runtime.output_dir)
    finally:
        scorer.close()
        encoder.close()
        proposer.unload()
    typer.echo(str(Path(config.runtime.output_dir).resolve()))


@app.command("validate-taxonomy")
def validate_taxonomy_command(
    taxonomy_path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    typer.echo(
        json.dumps(
            {"valid": True, "version": taxonomy.version, "labels": taxonomy.size},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@app.command("evaluate")
def evaluate_command(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    manifest_path: Optional[Path] = typer.Option(None, "--manifest", exists=True, dir_okay=False),
    output_path: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    cycles = read_jsonl(run_dir / "cycles.jsonl")
    result: dict = {}
    if any(row.get("metadata", {}).get("oracle_label") is not None for row in predictions):
        result.update(evaluate_prequential_predictions(predictions))
    if manifest_path is not None:
        result.update(evaluate_edit_events(cycles, read_json(manifest_path)))
    if not result:
        raise typer.BadParameter("No oracle labels were found and no manifest was supplied")
    destination = output_path or (run_dir / "metrics.json")
    atomic_write_json(destination, result)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


@app.command("generate-controlled")
def generate_controlled_command(
    event: str = typer.Option(..., "--event", help="add, split, prior-drift, covariate-drift, or anomaly"),
    data_path: Path = typer.Option(..., "--data", exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., "--output"),
    descriptions_path: Optional[Path] = typer.Option(None, "--descriptions", exists=True, dir_okay=False),
    anomaly_data_path: Optional[Path] = typer.Option(None, "--anomaly-data", exists=True, dir_okay=False),
    split_left: Optional[str] = typer.Option(None, "--split-left"),
    split_right: Optional[str] = typer.Option(None, "--split-right"),
    coarse_name: str = typer.Option("coarse category", "--coarse-name"),
    coarse_description: str = typer.Option("a coarse category containing two finer semantic modes", "--coarse-description"),
    seed: int = typer.Option(13, "--seed"),
    warmup: int = typer.Option(500, "--warmup"),
    post_length: int = typer.Option(1000, "--post-length"),
    horizon: int = typer.Option(256, "--horizon"),
) -> None:
    examples = load_labeled_examples(data_path)
    descriptions = load_label_descriptions(descriptions_path, [example.label for example in examples])
    normalized_event = event.strip().casefold().replace("_", "-")
    if normalized_event == "add":
        taxonomy, rows, manifest = generate_add_stream(
            examples,
            descriptions,
            seed=seed,
            warmup_per_class=max(1, warmup // max(1, len(set(e.label for e in examples)))),
            post_length=post_length,
            horizon=horizon,
        )
    elif normalized_event == "split":
        if split_left is None or split_right is None:
            raise typer.BadParameter("split requires --split-left and --split-right")
        taxonomy, rows, manifest = generate_split_stream(
            examples,
            descriptions,
            child_labels=(split_left, split_right),
            coarse_name=coarse_name,
            coarse_description=coarse_description,
            seed=seed,
            warmup_per_class=max(1, warmup // max(1, len(set(e.label for e in examples)))),
            post_length=post_length,
            horizon=horizon,
        )
    elif normalized_event == "prior-drift":
        taxonomy, rows, manifest = generate_prior_drift_stream(
            examples,
            descriptions,
            seed=seed,
            warmup_length=warmup,
            post_length=post_length,
            horizon=horizon,
        )
    elif normalized_event == "covariate-drift":
        taxonomy, rows, manifest = generate_covariate_drift_stream(
            examples,
            descriptions,
            seed=seed,
            warmup_length=warmup,
            post_length=post_length,
            horizon=horizon,
        )
    elif normalized_event == "anomaly":
        if anomaly_data_path is None:
            raise typer.BadParameter("anomaly requires --anomaly-data")
        anomaly_examples = load_labeled_examples(anomaly_data_path)
        taxonomy, rows, manifest = generate_anomaly_stream(
            examples,
            anomaly_examples,
            descriptions,
            seed=seed,
            warmup_length=warmup,
            post_length=post_length,
            horizon=horizon,
        )
    else:
        raise typer.BadParameter(f"Unsupported event type: {event}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "initial_taxonomy.json", taxonomy.to_dict())
    write_jsonl(output_dir / "stream.jsonl", rows)
    atomic_write_json(output_dir / "manifest.json", manifest)
    typer.echo(str(output_dir.resolve()))
