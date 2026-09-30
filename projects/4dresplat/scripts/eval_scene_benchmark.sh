#!/usr/bin/env bash
# ============================================================
# eval_scene_benchmark.sh
# Generic per-scene benchmark evaluation driver.
#
# Usage:
#   bash scripts/eval_scene_benchmark.sh [SCENE]
#   SCENE  Any scene name under benchmark/scenes/ (default: americano)
#          americano | coffee_martini | cook_spinach | cut_lemon |
#          cut_roasted_beef | espresso | flame_salmon | flame_steak |
#          keyboard | sear_steak | split_cookie | torchchocolate
#
# Environment variables:
#   EVAL_QUERY_IDS    Comma/space-separated query_id subset
#   EVAL_QUERY_LIMIT  Evaluate only the first N queries
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/common.sh"

SCENE="${1:-americano}"

# --- Scene path mapping ---
case "${SCENE}" in
    split_cookie)    RUN_SUB="split-cookie"; DS_SUB="split-cookie"  ;;
    torchchocolate)  RUN_SUB="torchocolate"; DS_SUB="torchocolate"  ;;
    americano|coffee_martini|cook_spinach|cut_lemon|cut_roasted_beef|\
    espresso|flame_salmon|flame_steak|keyboard|sear_steak)
        RUN_SUB="${SCENE}"; DS_SUB="${SCENE}" ;;
    *)
        if [[ ! -d "${GS_ROOT}/benchmark/scenes/${SCENE}" ]]; then
            echo "ERROR: Unknown scene '${SCENE}' (not under benchmark/scenes/)"
            echo "       Available: $(ls "${GS_ROOT}/benchmark/scenes/" | tr '\n' ' ')"
            exit 1
        fi
        RUN_SUB="${SCENE}"; DS_SUB="${SCENE}"
        ;;
esac

RUN_DIR="${GS_ROOT}/runs/plain4dgs_4k/hypernerf/${RUN_SUB}"
DATASET_DIR="${GS_ROOT}/data/hypernerf/misc/${DS_SUB}"
BENCHMARK_DIR="${GS_ROOT}/benchmark"
REPORT_DIR="${GS_ROOT}/reports/${SCENE}_eval"

BENCHMARK_JSON="${REPORT_DIR}/${SCENE}_benchmark.json"
QUERY_ROOT_MAP="${REPORT_DIR}/query_root_map.json"
DATASET_DIR_MAP="${REPORT_DIR}/dataset_dir_map.json"
EVAL_JSON="${REPORT_DIR}/eval_results.json"
EVAL_MD="${REPORT_DIR}/eval_results.md"

mkdir -p "${REPORT_DIR}"

echo "=== Step 1: Build merged benchmark JSON (Questions + Answers) for ${SCENE} ==="
gs_python "${SCRIPT_DIR}/build_benchmark_json.py" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --scenes "${SCENE}" \
    --output "${BENCHMARK_JSON}"

echo ""
echo "=== Step 2: Build query_root_map.json ==="
export GS_ROOT RUN_DIR DATASET_DIR REPORT_DIR BENCHMARK_JSON
python3 - <<'PYEOF'
import json, os
from pathlib import Path

GS_ROOT        = Path(os.environ["GS_ROOT"])
RUN_DIR        = Path(os.environ["RUN_DIR"])
DATASET_DIR    = Path(os.environ["DATASET_DIR"])
REPORT_DIR     = Path(os.environ["REPORT_DIR"])
BENCHMARK_JSON = Path(os.environ["BENCHMARK_JSON"])

data = json.loads(BENCHMARK_JSON.read_text())
qs = data.get("queries", data) if isinstance(data, dict) else data
if isinstance(qs, dict):
    qs = list(qs.values())
query_ids_raw = os.environ.get("EVAL_QUERY_IDS", "").strip()
limit_raw = os.environ.get("EVAL_QUERY_LIMIT", "").strip()
if query_ids_raw:
    keep = {item.strip() for item in query_ids_raw.replace(",", " ").split() if item.strip()}
    qs = [q for q in qs if str(q.get("query_id", q.get("id"))) in keep]
if limit_raw:
    qs = qs[: max(int(limit_raw), 0)]
if isinstance(data, dict) and "queries" in data:
    data["queries"] = qs
    BENCHMARK_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
else:
    BENCHMARK_JSON.write_text(json.dumps(qs, indent=2, ensure_ascii=False), encoding="utf-8")

root_map = {}
ds_map   = {}
for q in qs:
    qid = str(q.get("query_id", q.get("id")))
    if not qid or qid == "None": continue
    q_dir = RUN_DIR / "entitybank" / "query_guided" / qid
    root_map[qid] = str(q_dir)
    ds_map[qid]   = str(DATASET_DIR)

out1 = REPORT_DIR / "query_root_map.json"
out2 = REPORT_DIR / "dataset_dir_map.json"
out1.write_text(json.dumps(root_map, indent=2))
out2.write_text(json.dumps(ds_map, indent=2))

ready = 0
fast_ready = 0
for qid in root_map:
    qroot = Path(root_map[qid])
    has_validation = (qroot / "final_query_render_sourcebg" / "validation.json").exists()
    has_sam3_store = (qroot / "sam3_tracks" / "sam3_query_tracks.json").exists()
    if has_validation or has_sam3_store:
        ready += 1
    if (not has_validation) and has_sam3_store:
        fast_ready += 1
print(f"Ready: {ready}/{len(root_map)}  (sam3-fast-only: {fast_ready})")
PYEOF

echo ""
echo "=== Step 3: Run evaluation ==="
gs_python "${SCRIPT_DIR}/evaluate_ours_benchmark.py" \
    --benchmark "${BENCHMARK_JSON}" \
    --benchmark-root "${BENCHMARK_DIR}" \
    --query-root-map "${QUERY_ROOT_MAP}" \
    --dataset-dir-map "${DATASET_DIR_MAP}" \
    --output-json "${EVAL_JSON}" \
    --output-md "${EVAL_MD}" \
    --skip-missing

echo ""
echo "=== Results ==="
cat "${EVAL_MD}" 2>/dev/null || true
