# Implementation conventions

## Method settings

The default configuration uses `alpha=0.05`, `B=64`, `R=4` including `stay`, `H=256`, the specified background description and label-conditioning template, Qwen/Llama evidence backbones, BGE-M3 residual embeddings, Qwen proposer models, greedy decoding, 512 output tokens, 8-token names, 48-token descriptions, add/split edit semantics, prior transforms, future-only evidence, the `1/[j(j+1)]` spending schedule, the floor-aware boundary, common-support sequence termination, the standard SVD rank cutoff, repeated-eigenvalue canonicalization, arrival/hash tie-breaking, and candidate-identifier tie-breaking.

## Operational conventions

1. **Proposer prompt.** The complete prompt is versioned in `src/sbls/proposer.py` and enforces the input fields, output grammar, decoding constraints, length limits, and parser requirements.
2. **Truncation representation.** The configuration accepts an explicit token identifier. When it is unset, the configured text `<trunc>` is tokenized without modifying the frozen model vocabulary.
3. **Dataset manifests.** Stream preparation uses explicit event manifests. The BOLT converter normalizes supported input structures and writes the canonical examples, descriptions, and preparation metadata used by the controlled-stream generators.
4. **Semantic-edit evaluation.** The evaluator supports acceptable generated-name sets and disjoint audit records for Hungarian label alignment. The selected mapping scope is reported in the evaluation output.

These conventions are deterministic, configuration-controlled where applicable, and recorded in the run outputs.
