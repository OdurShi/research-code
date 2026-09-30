#!/usr/bin/env bash
# ============================================================
# run_scene_queries.sh — high-throughput query pipeline (~60-100 queries/scene)
#
# Phases (fewer Qwen reloads + semantic export stamp cache + parallel render):
#
#   Phase 1  Batch Step 3a: all-query Qwen planning (single model load)
#   Phase 2  Per-entity:    entity cache + 3b+3c (MISS entities only)
#   Phase 3  Per-query:     3c+ entitybank + 3c++ semantic exports (stamp cache)
#   Phase 4  Batch Step 3d-pre: all-query Qwen semantic assignment (single load)
#   Phase 5  Parallel 3d+3e:   concurrent selection + render (CPU/IO bound)
#
# Usage:
#   bash scripts/run_scene_queries.sh \
#       --run-dir  runs/d4resplat/hypernerf/SCENE \
#       --dataset  data/hypernerf/misc/SCENE \
#       --queries  benchmark/scenes/SCENE/queries.json \
#       [--gpu 0] [--workers 4] [--dry-run]
#
# queries.json format (auto-generated from benchmark):
#   [{"query_id":"americano_q001","query_text":"The hand pouring coffee."},...]
#
# Logs: reports/runs/<scene>/<timestamp>/
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Argument parsing ---
RUN_DIR=""
DATASET_DIR=""
QUERIES_JSON=""
GPU=0
WORKERS=4
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)   RUN_DIR="$2";      shift 2 ;;
        --dataset)   DATASET_DIR="$2";  shift 2 ;;
        --queries)   QUERIES_JSON="$2"; shift 2 ;;
        --gpu)       GPU="$2";          shift 2 ;;
        --workers)   WORKERS="$2";      shift 2 ;;
        --dry-run)   DRY_RUN=1;         shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "${RUN_DIR}" || -z "${DATASET_DIR}" || -z "${QUERIES_JSON}" ]] && {
    echo "Usage: $0 --run-dir DIR --dataset DIR --queries JSON [--gpu N] [--workers N]"
    exit 1
}

SCENE_NAME="$(basename "${RUN_DIR}")"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export GSAM2_FRAME_SUBSAMPLE_STRIDE="${GSAM2_FRAME_SUBSAMPLE_STRIDE:-32}"
export GSAM2_NUM_CONTEXT_FRAMES="${GSAM2_NUM_CONTEXT_FRAMES:-16}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

CONDA_ENV="/root/autodl-tmp/.conda-envs/gs4d-cuda121-py310"
CONDA_RUN="conda run --no-capture-output -p ${CONDA_ENV}"

# --- Logging ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${GS_ROOT}/reports/runs/${SCENE_NAME}/${TIMESTAMP}"
mkdir -p "${LOG_ROOT}"
LOGFILE="${LOG_ROOT}/pipeline.log"
MANIFEST_DIR="${LOG_ROOT}/manifests"
mkdir -p "${MANIFEST_DIR}"

log() { local msg="[$(date '+%H:%M:%S')] $*"; echo "${msg}" | tee -a "${LOGFILE}"; }
log_path() {
    local label="$1" path="$2"
    [[ -e "${path}" ]] && echo "    ${label}: ${path}  [✓]" | tee -a "${LOGFILE}" \
                       || echo "    ${label}: ${path}  [·]" | tee -a "${LOGFILE}"
}
step_start() { log "━━━ STEP $1: $2 ━━━"; STEP_T0="${SECONDS}"; }
step_done()  { log "    ✓ done in $(( SECONDS - STEP_T0 ))s"; }

# --- Preconditions ---
[[ -d "${RUN_DIR}/point_cloud" ]] || { echo "ERROR: ${RUN_DIR}/point_cloud missing"; exit 1; }
[[ -f "${QUERIES_JSON}" ]] || { echo "ERROR: ${QUERIES_JSON} missing"; exit 1; }

CACHE_DIR="${RUN_DIR}/entitybank/query_cache"

log "======================================================="
log " run_scene_queries — ${SCENE_NAME}"
log " RUN_DIR    : ${RUN_DIR}"
log " DATASET_DIR: ${DATASET_DIR}"
log " QUERIES    : ${QUERIES_JSON}"
log " GPU        : ${GPU}  WORKERS: ${WORKERS}"
log " LOGFILE    : ${LOGFILE}"
log "======================================================="

# --- Load query list ---
QUERY_IDS_CSV=$(python3 -c "
import json, sys
qs = json.load(open('${QUERIES_JSON}'))
print(','.join(q['query_id'] for q in qs))
")
IFS=',' read -ra QUERY_IDS <<< "${QUERY_IDS_CSV}"
N_QUERIES="${#QUERY_IDS[@]}"
log " ${N_QUERIES} queries loaded"
log ""

PIPELINE_T0="${SECONDS}"

# ════════════════════════════════════════════════════════════
# Phase 1: Batch Step 3a — all-query Qwen planning (single model load)
# ════════════════════════════════════════════════════════════
step_start "Phase1" "Batch Qwen Query Planning (Step 3a, single load)"

PLAN_MANIFEST="${MANIFEST_DIR}/phase1_plan_manifest.json"
python3 - <<PYEOF
import json
from pathlib import Path

qs = json.load(open('${QUERIES_JSON}'))
run_dir = Path('${RUN_DIR}')
dataset_dir = '${DATASET_DIR}'
items = []
for q in qs:
    qid = q['query_id']
    out_dir = run_dir / 'entitybank' / 'query_guided' / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    items.append({
        'query_id': qid,
        'query_text': q['query_text'],
        'dataset_dir': dataset_dir,
        'output_path': str(out_dir / 'query_plan.json'),
    })
Path('${PLAN_MANIFEST}').write_text(json.dumps(items, indent=2, ensure_ascii=False))
print(f"[phase1] manifest: {len(items)} queries → ${PLAN_MANIFEST}")
PYEOF

if [[ "${DRY_RUN}" -eq 0 ]]; then
    ${CONDA_RUN} python "${SCRIPT_DIR}/batch_plan_queries.py" \
        --manifest "${PLAN_MANIFEST}" \
        --coarse-stride "${GSAM2_FRAME_SUBSAMPLE_STRIDE}" \
        --fine-frames   "${GSAM2_NUM_CONTEXT_FRAMES}" \
        --summary-path  "${LOG_ROOT}/phase1_summary.json" \
        2>&1 | tee -a "${LOGFILE}" | grep -E "subject|time_window|done|error|ERROR|Traceback|batch_plan" || true
fi
step_done

# ════════════════════════════════════════════════════════════
# Phase 2: Per-entity — entity cache + 3b+3c (MISS entities only)
# ════════════════════════════════════════════════════════════
step_start "Phase2" "Per-entity 3b+3c (cache-aware)"

# Collect unique subjects; run SAM3 only on cache MISS
ENTITY_JOBS="${MANIFEST_DIR}/phase2_entity_jobs.json"
python3 - <<PYEOF
import json, re
from pathlib import Path

qs = json.load(open('${QUERIES_JSON}'))
run_dir = Path('${RUN_DIR}')
jobs = []
seen_subjects = {}

for q in qs:
    qid = q['query_id']
    plan_path = run_dir / 'entitybank' / 'query_guided' / qid / 'query_plan.json'
    if not plan_path.exists():
        print(f"[phase2] WARN: no plan for {qid}, skip")
        continue
    plan = json.load(open(plan_path))
    subject = plan.get('subject', '').strip()
    if not subject:
        phrases = plan.get('query_subject_phrases') or []
        subject = phrases[0].strip() if phrases else 'object'
    time_window = plan.get('time_window') or {}
    jobs.append({
        'query_id': qid,
        'query_text': q['query_text'],
        'subject': subject,
        'time_window': time_window,
        'plan_path': str(plan_path),
        'output_dir': str(run_dir / 'entitybank' / 'query_guided' / qid),
    })

Path('${ENTITY_JOBS}').write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
print(f"[phase2] {len(jobs)} query jobs prepared")
PYEOF

# Per-query cache check; run 3b+3c only on MISS
while IFS= read -r LINE; do
    QUERY_ID=$(echo "${LINE}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['query_id'])")
    QUERY_TEXT=$(echo "${LINE}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['query_text'])")
    SUBJECT=$(echo "${LINE}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['subject'])")
    TW=$(echo "${LINE}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(json.dumps(d['time_window']))")
    PLAN_PATH=$(echo "${LINE}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['plan_path'])")
    Q_OUTPUT_DIR=$(echo "${LINE}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['output_dir'])")

    TRACK_DIR="${Q_OUTPUT_DIR}/grounded_sam2"
    TRACKS_PATH="${TRACK_DIR}/grounded_sam2_query_tracks.json"
    PROPOSAL_DIR="${Q_OUTPUT_DIR}/proposal_dir"
    Q_LOG="${LOG_ROOT}/${QUERY_ID}.log"

    log "  [phase2] ${QUERY_ID}: subject='${SUBJECT}'"

    # Cache lookup
    CACHE_RESULT=$(${CONDA_RUN} python "${SCRIPT_DIR}/entity_cache_helper.py" lookup \
        --cache-dir "${CACHE_DIR}" \
        --subject "${SUBJECT}" \
        --dataset-dir "${DATASET_DIR}" \
        --time-window "${TW}" \
        2>&1 || true)
    CACHE_HIT_DIR=$(echo "${CACHE_RESULT}" | grep "^HIT " | sed 's/^HIT //' || true)

    if [[ -n "${CACHE_HIT_DIR}" ]] && [[ -d "${CACHE_HIT_DIR}" ]]; then
        log "    [CACHE HIT] → ${CACHE_HIT_DIR}"
        # Write symlink info so Phase 3 can find proposal_dir
        python3 -c "
import json; from pathlib import Path
info_path = Path('${Q_OUTPUT_DIR}') / '.proposal_dir_ref'
info_path.write_text('${CACHE_HIT_DIR}')
"
    else
        log "    [CACHE MISS] running 3b+3c"

        if [[ "${DRY_RUN}" -eq 0 ]]; then
            # 3b: SAM3 tracking
            if [[ ! -f "${TRACKS_PATH}" ]]; then
                ${CONDA_RUN} python "${SCRIPT_DIR}/run_grounded_sam2_query.py" \
                    --dataset-dir "${DATASET_DIR}" \
                    --query-plan-path "${PLAN_PATH}" \
                    --output-dir "${TRACK_DIR}" \
                    --sam2-model-id "${GS_ROOT}/sam3.1/sam3.1_multiplex.pt" \
                    --grounding-model-id "IDEA-Research/grounding-dino-base" \
                    --prompt-type point \
                    --detector-frame-stride 6 \
                    --max-detector-frames 48 \
                    --detection-top-k 5 \
                    --box-threshold 0.25 \
                    --text-threshold 0.20 \
                    --num-point-prompts 16 \
                    --track-window-radius 160 \
                    --frame-subsample-stride "${GSAM2_FRAME_SUBSAMPLE_STRIDE}" \
                    --num-anchor-seeds 3 \
                    2>&1 | tee -a "${Q_LOG}" | grep -E "\[track\]|\[det\]|Saved|error|Error|Traceback" || true
            fi

            # 3c: Gaussian proposal
            if [[ ! -f "${PROPOSAL_DIR}/assignments.json" ]]; then
                ${CONDA_RUN} python "${SCRIPT_DIR}/build_query_proposal_dir.py" \
                    --run-dir "${RUN_DIR}" \
                    --dataset-dir "${DATASET_DIR}" \
                    --tracks-path "${TRACKS_PATH}" \
                    --output-dir "${PROPOSAL_DIR}" \
                    --max-track-frames 16 \
                    --proposal-keep-ratio 0.03 \
                    --min-gaussians 256 \
                    --max-gaussians 4096 \
                    2>&1 | tee -a "${Q_LOG}" | grep -E "proposal|entity|Done|error|Error|Traceback" || true
            fi

            # Persist cache entry
            if [[ -f "${PROPOSAL_DIR}/assignments.json" ]]; then
                ${CONDA_RUN} python "${SCRIPT_DIR}/entity_cache_helper.py" store \
                    --cache-dir "${CACHE_DIR}" \
                    --subject "${SUBJECT}" \
                    --proposal-dir "${PROPOSAL_DIR}" \
                    --dataset-dir "${DATASET_DIR}" \
                    --time-window "${TW}" \
                    2>&1 | tee -a "${Q_LOG}" || true
                python3 -c "
from pathlib import Path
(Path('${Q_OUTPUT_DIR}') / '.proposal_dir_ref').write_text('${PROPOSAL_DIR}')
"
            fi
        fi
    fi
done < <(python3 -c "
import json, sys
jobs = json.load(open('${ENTITY_JOBS}'))
for j in jobs:
    print(json.dumps(j))
")

step_done

# ════════════════════════════════════════════════════════════
# Phase 3: Per-query — 3c+ entitybank + 3c++ semantic exports (stamp cache)
# ════════════════════════════════════════════════════════════
step_start "Phase3" "Per-query entitybank + semantic exports"

for QUERY_ID in "${QUERY_IDS[@]}"; do
    Q_OUTPUT_DIR="${RUN_DIR}/entitybank/query_guided/${QUERY_ID}"
    PROPOSAL_REF="${Q_OUTPUT_DIR}/.proposal_dir_ref"
    Q_LOG="${LOG_ROOT}/${QUERY_ID}.log"

    # Resolve proposal dir
    if [[ -f "${PROPOSAL_REF}" ]]; then
        PROPOSAL_DIR="$(cat "${PROPOSAL_REF}")"
    else
        PROPOSAL_DIR="${Q_OUTPUT_DIR}/proposal_dir"
    fi

    QUERY_ENTITYBANK_DIR="${Q_OUTPUT_DIR}/query_entitybank"
    QUERY_RUN_DIR="${Q_OUTPUT_DIR}/query_worldtube_run"

    log "  [phase3] ${QUERY_ID} proposal=${PROPOSAL_DIR}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then continue; fi

    # 3c+: export entitybank
    if [[ ! -f "${QUERY_ENTITYBANK_DIR}/entities.json" ]]; then
        ${CONDA_RUN} python "${SCRIPT_DIR}/export_entitybank.py" \
            --run-dir "${RUN_DIR}" \
            --proposal-dir "${PROPOSAL_DIR}" \
            --proposal-strict \
            --output-dir "${QUERY_ENTITYBANK_DIR}" \
            --max-entities 12 \
            --min-gaussians-per-entity 32 \
            2>&1 | tee -a "${Q_LOG}" | grep -E "Done|entity|saved|error|Error|Traceback" || true
    fi

    # Build query_worldtube_run symlinks
    mkdir -p "${QUERY_RUN_DIR}"
    ln -sfn "${RUN_DIR}/config.yaml"  "${QUERY_RUN_DIR}/config.yaml"
    ln -sfn "${RUN_DIR}/point_cloud"  "${QUERY_RUN_DIR}/point_cloud"
    ln -sfn "${RUN_DIR}/test"         "${QUERY_RUN_DIR}/test"
    ln -sfn "${QUERY_ENTITYBANK_DIR}" "${QUERY_RUN_DIR}/entitybank"

    # 3c++: semantic exports (stamp cache)
    EXPORT_KEY=$(echo "${PROPOSAL_DIR}" | md5sum | cut -c1-8)
    SEMANTIC_STAMP="${QUERY_RUN_DIR}/entitybank/.semantic_export_${EXPORT_KEY}"
    if [[ ! -f "${SEMANTIC_STAMP}" ]]; then
        for EXPORT_SCRIPT in export_semantic_slots export_semantic_tracks export_semantic_priors export_native_semantics; do
            ${CONDA_RUN} python "${SCRIPT_DIR}/${EXPORT_SCRIPT}.py" \
                --run-dir "${QUERY_RUN_DIR}" \
                2>&1 | tee -a "${Q_LOG}" | grep -E "saved|Done|error|Error|Traceback" || true
        done
        touch "${SEMANTIC_STAMP}"
        log "    [${QUERY_ID}] 3c++ done (stamp=${EXPORT_KEY})"
    else
        log "    [${QUERY_ID}] 3c++ SKIP (stamp ${EXPORT_KEY})"
    fi
done

step_done

# ════════════════════════════════════════════════════════════
# Phase 4: Batch Step 3d-pre — all-query Qwen semantic assignment (single load)
# ════════════════════════════════════════════════════════════
step_start "Phase4" "Batch Qwen Semantic Assignments (Step 3d-pre, single load)"

SEM_MANIFEST="${MANIFEST_DIR}/phase4_sem_manifest.json"
python3 - <<PYEOF
import json
from pathlib import Path

qs = json.load(open('${QUERIES_JSON}'))
run_dir = Path('${RUN_DIR}')
items = []
for q in qs:
    qid = q['query_id']
    q_run = run_dir / 'entitybank' / 'query_guided' / qid / 'query_worldtube_run'
    out_path = q_run / 'entitybank' / 'semantic_assignments_qwen.json'
    items.append({
        'query_id':   qid,
        'run_dir':    str(q_run),
        'query_text': q['query_text'],
        'output_path': str(out_path),
    })
Path('${SEM_MANIFEST}').write_text(json.dumps(items, indent=2, ensure_ascii=False))
print(f"[phase4] manifest: {len(items)} queries → ${SEM_MANIFEST}")
PYEOF

if [[ "${DRY_RUN}" -eq 0 ]]; then
    ${CONDA_RUN} python "${SCRIPT_DIR}/batch_export_qwen_semantics.py" \
        --manifest "${SEM_MANIFEST}" \
        --max-entities 12 \
        --summary-path "${LOG_ROOT}/phase4_summary.json" \
        2>&1 | tee -a "${LOGFILE}" | grep -E "saved|done|error|ERROR|Traceback|batch_qwen_sem" || true
fi

step_done

# ════════════════════════════════════════════════════════════
# Phase 5: Parallel 3d + 3e — concurrent selection + render
# ════════════════════════════════════════════════════════════
step_start "Phase5" "Parallel 3d + 3e (workers=${WORKERS})"

run_3d_3e() {
    local QUERY_ID="$1"
    local QUERY_TEXT="$2"
    local Q_OUTPUT_DIR="${RUN_DIR}/entitybank/query_guided/${QUERY_ID}"
    local PLAN_PATH="${Q_OUTPUT_DIR}/query_plan.json"
    local QUERY_RUN_DIR="${Q_OUTPUT_DIR}/query_worldtube_run"
    local QWEN_ASSIGNMENTS="${QUERY_RUN_DIR}/entitybank/semantic_assignments_qwen.json"
    local QWEN_SELECTION="${QUERY_RUN_DIR}/entitybank/selected_query_qwen.json"
    local FINAL_RENDER_DIR="${Q_OUTPUT_DIR}/final_query_render_sourcebg"
    local Q_LOG="${LOG_ROOT}/${QUERY_ID}.log"

    # 3d: entity selection
    if [[ ! -f "${QWEN_SELECTION}" ]]; then
        ${CONDA_RUN} python "${SCRIPT_DIR}/select_qwen_query_entities.py" \
            --assignments-path "${QWEN_ASSIGNMENTS}" \
            --query "${QUERY_TEXT}" \
            --query-plan-path "${PLAN_PATH}" \
            --output-path "${QWEN_SELECTION}" \
            2>&1 | tee -a "${Q_LOG}" | grep -E "selected|entity|Done|error|Error|Traceback" || true
    fi

    # 3e: render
    if [[ ! -f "${FINAL_RENDER_DIR}/validation.json" ]]; then
        ${CONDA_RUN} python "${SCRIPT_DIR}/render_query_video.py" \
            --run-dir "${QUERY_RUN_DIR}" \
            --dataset-dir "${DATASET_DIR}" \
            --selection-path "${QWEN_SELECTION}" \
            --output-dir "${FINAL_RENDER_DIR}" \
            --background-mode source \
            --fps 6 \
            --stride 1 \
            2>&1 | tee -a "${Q_LOG}" | grep -E "frame|render|saved|Done|error|Error|Traceback" || true
    fi

    # Status
    [[ -f "${FINAL_RENDER_DIR}/validation.json" ]] && \
        echo "[phase5] ${QUERY_ID}: ✓ done" || \
        echo "[phase5] ${QUERY_ID}: ✗ validation.json missing"
}
export -f run_3d_3e
export RUN_DIR DATASET_DIR LOG_ROOT SCRIPT_DIR CONDA_RUN

if [[ "${DRY_RUN}" -eq 0 ]]; then
    python3 -c "
import json
qs = json.load(open('${QUERIES_JSON}'))
for q in qs:
    print(q['query_id'] + '\t' + q['query_text'])
" | xargs -P "${WORKERS}" -I{} bash -c '
    IFS=$'"'"'\t'"'"' read -r QID QTEXT <<< "{}"
    run_3d_3e "${QID}" "${QTEXT}" 2>&1 | tee -a "${LOG_ROOT}/${QID}.log"
' 2>&1 | tee -a "${LOGFILE}" | grep -E "\[phase5\]" || true
fi

step_done

# --- Summary ---
TOTAL=$(( SECONDS - PIPELINE_T0 ))
DONE_COUNT=$(python3 -c "
import json; from pathlib import Path
qs = json.load(open('${QUERIES_JSON}'))
run_dir = Path('${RUN_DIR}')
done = sum(1 for q in qs
           if (run_dir / 'entitybank' / 'query_guided' / q['query_id']
               / 'final_query_render_sourcebg' / 'validation.json').exists())
print(done)
")
log "======================================================="
log " Finished ${DONE_COUNT}/${N_QUERIES} queries in ${TOTAL}s ($(( TOTAL/60 ))m$(( TOTAL%60 ))s)"
log " Log directory: ${LOG_ROOT}"
log " Next: bash scripts/eval_scene.sh --run-dir ${RUN_DIR} --queries ${QUERIES_JSON}"
log "======================================================="
