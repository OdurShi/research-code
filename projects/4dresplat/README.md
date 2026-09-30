# 4DReSplat

Tools for language-guided segmentation of dynamic Gaussian scenes. The pipeline combines query planning, text-guided tracking, entity selection, and rendering.

## Model Weights

### Qwen3-VL-8B-Instruct (Query Planner)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='models/Qwen3-VL-8B-Instruct')
"
```

Default path: `models/Qwen3-VL-8B-Instruct/`. Override with:

```bash
export D4RESPLAT_QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

### SAM3.1 (Text-guided Tracking)

```bash
modelscope download --model facebook/sam3.1
```

## Running

### Full scene pipeline 

```bash
QWEN_BATCH_SIZE=50 bash scripts/run_scene_benchmark.sh scene_name (e.g. americano)
```

### Evaluation

```bash
bash scripts/eval_scene_benchmark.sh scene_name (e.g. americano)
```

## Repository Layout

```
Reason4D/
├── d4resplat/             # 4DReSplat core library (Python package)
│   ├── temporal/          # 4D Gaussian reconstruction 
│   ├── entitybank/        # entity-centric scene memory 
│   └── semantics/         # query planning, tracking, scoring, and grounding
├── scripts/               # training, evaluation, and data preparation
├── external/              # 4DGaussians submodule
├── data/                  # datasets
└── runs/                  # experiment outputs
```


