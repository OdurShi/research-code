#!/usr/bin/env bash
# ============================================================
# run_scene_benchmark.sh
# Full-scene benchmark pipeline for every entry under benchmark/scenes/.
#
# Usage:
#   bash scripts/run_scene_benchmark.sh [SCENE] [GPU_ID]
#   SCENE  Any scene name under benchmark/scenes/ (default: americano)
#          americano | coffee_martini | cook_spinach | cut_lemon |
#          cut_roasted_beef | espresso | flame_salmon | flame_steak |
#          keyboard | sear_steak | split_cookie | torchchocolate
#   GPU_ID  default: 0
#
# Environment variables (same as run_americano_all.sh):
#   RUN_QUERY_IDS      Comma/space-separated query_id subset
#   RUN_QUERY_LIMIT    Run at most the first N queries
#   SAM3_NUM_PROCS     SAM3.1 parallel processes (default 1)
#   SAM3_FRAME_STRIDE  SAM3 frame stride (default 20)
#   SCENE_RENDER_JOBS  Parallel render jobs (default 1)
#   RUN_SKIP_3E        Set to 1 to skip render step
#   QUICK_EVAL_INTERVAL Quick-eval every N queries (default 4)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/common.sh"

SCENE="${1:-americano}"
GPU="${2:-0}"

# --- Scene path mapping ---
# RUN_SUB = directory name under runs/plain4dgs_4k/hypernerf/
#       DS_SUB  = directory name under data/hypernerf/misc/
# Usually RUN_SUB/DS_SUB match the benchmark key; a few aliases are listed.
case "${SCENE}" in
    split_cookie)    RUN_SUB="split-cookie"; DS_SUB="split-cookie"  ;;
    torchchocolate)  RUN_SUB="torchocolate"; DS_SUB="torchocolate"  ;;
    americano|coffee_martini|cook_spinach|cut_lemon|cut_roasted_beef|\
    espresso|flame_salmon|flame_steak|keyboard|sear_steak)
        RUN_SUB="${SCENE}"; DS_SUB="${SCENE}" ;;
    *)
        # Ensure benchmark/scenes contains this scene
        if [[ ! -d "${GS_ROOT}/benchmark/scenes/${SCENE}" ]]; then
            echo "ERROR: Unknown scene '${SCENE}' (missing under benchmark/scenes/)"
            echo "       Available scenes: $(ls "${GS_ROOT}/benchmark/scenes/" | tr '\n' ' ')"
            exit 1
        fi
        RUN_SUB="${SCENE}"; DS_SUB="${SCENE}"
        ;;
esac

RUN_DIR="${GS_ROOT}/runs/plain4dgs_4k/hypernerf/${RUN_SUB}"
DATASET_DIR="${GS_ROOT}/data/hypernerf/misc/${DS_SUB}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export GSAM2_FRAME_SUBSAMPLE_STRIDE=32
export GSAM2_NUM_CONTEXT_FRAMES=8
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GS_ROOT}:${PYTHONPATH:-}"
export QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-1}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

# --- Logging ---
LOG_ROOT="${GS_ROOT}/reports/runs/${SCENE}_benchmark"
mkdir -p "${LOG_ROOT}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOG_ROOT}/run_${TIMESTAMP}.log"
TIMING_JSONL="${LOG_ROOT}/timing_${TIMESTAMP}.jsonl"
TIMING_SUMMARY_MD="${LOG_ROOT}/timing_${TIMESTAMP}.md"
TIMING_SUMMARY_JSON="${LOG_ROOT}/timing_${TIMESTAMP}.json"
ACTIVE_QUERY="__pipeline__"

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "${msg}" | tee -a "${LOGFILE}"
}

timing_record() {
    local query_id="$1"
    local step_id="$2"
    local phase="$3"
    local elapsed_s="$4"
    local status="$5"
    local detail="${6:-{}}"
    python3 - "${TIMING_JSONL}" "${query_id}" "${step_id}" "${phase}" "${elapsed_s}" "${status}" "${detail}" <<'PYEOF'
import datetime as dt
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
detail_text = sys.argv[7] if len(sys.argv) > 7 else "{}"
try:
    detail = json.loads(detail_text)
except Exception:
    detail = {"note": detail_text}
payload = {
    "ts_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    "query_id": sys.argv[2],
    "step_id": sys.argv[3],
    "phase": sys.argv[4],
    "elapsed_s": int(float(sys.argv[5])),
    "status": sys.argv[6],
    "detail": detail,
}
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PYEOF
}

probe_path_json() {
    local target_path="$1"
    python3 - "${target_path}" <<'PYEOF'
import json
import os
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
out = {"path": str(p), "exists": p.exists(), "files": 0, "bytes": 0}
if p.exists():
    if p.is_file():
        out["files"] = 1
        out["bytes"] = int(p.stat().st_size)
    elif p.is_dir():
        files = 0
        total_bytes = 0
        for root, _, names in os.walk(p):
            for name in names:
                files += 1
                fp = os.path.join(root, name)
                try:
                    total_bytes += int(os.path.getsize(fp))
                except Exception:
                    pass
        out["files"] = files
        out["bytes"] = total_bytes
print(json.dumps(out, ensure_ascii=False))
PYEOF
}

timing_summarize() {
    python3 - "${TIMING_JSONL}" "${TIMING_SUMMARY_MD}" "${TIMING_SUMMARY_JSON}" "${SCENE}" <<'PYEOF'
import json
import pathlib
import sys

jsonl_path = pathlib.Path(sys.argv[1])
md_path = pathlib.Path(sys.argv[2])
summary_path = pathlib.Path(sys.argv[3])
scene = sys.argv[4]

rows = []
if jsonl_path.exists():
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

phase_rows = [r for r in rows if r.get("status") == "done" and r.get("step_id") != "query_total"]
query_rows = [r for r in rows if r.get("step_id") == "query_total" and r.get("status") == "done"]

phase_agg = {}
for r in phase_rows:
    key = (r.get("step_id", ""), r.get("phase", ""))
    phase_agg.setdefault(key, []).append(int(r.get("elapsed_s", 0)))

phase_summary = []
for key, vals in sorted(phase_agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    phase_summary.append({
        "step_id": key[0],
        "phase": key[1],
        "count": len(vals),
        "total_s": int(sum(vals)),
        "avg_s": round(sum(vals) / len(vals), 2),
        "max_s": int(max(vals)),
        "p95_s": int(sorted(vals)[max(0, int(len(vals) * 0.95) - 1)]),
    })

query_summary = []
for r in query_rows:
    query_summary.append({
        "query_id": r.get("query_id", ""),
        "elapsed_s": int(r.get("elapsed_s", 0)),
    })
query_summary.sort(key=lambda x: x["elapsed_s"], reverse=True)

payload = {
    "records_total": len(rows),
    "records_done": sum(1 for r in rows if r.get("status") == "done"),
    "records_failed": sum(1 for r in rows if r.get("status") == "failed"),
    "phase_summary": phase_summary,
    "slowest_queries": query_summary[:20],
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = []
lines.append(f"# {scene} benchmark timing summary")
lines.append("")
lines.append(f"- raw jsonl: `{jsonl_path}`")
lines.append(f"- summary json: `{summary_path}`")
lines.append(f"- done records: `{payload['records_done']}` / total `{payload['records_total']}`")
lines.append("")
lines.append("## Phase Aggregation")
lines.append("")
lines.append("| step | phase | count | total(s) | avg(s) | p95(s) | max(s) |")
lines.append("|---|---|---:|---:|---:|---:|---:|")
for row in phase_summary:
    lines.append(
        f"| {row['step_id']} | {row['phase']} | {row['count']} | {row['total_s']} | {row['avg_s']} | {row['p95_s']} | {row['max_s']} |"
    )
lines.append("")
lines.append("## Slowest Queries")
lines.append("")
lines.append("| query_id | total(s) |")
lines.append("|---|---:|")
for row in query_summary[:20]:
    lines.append(f"| {row['query_id']} | {row['elapsed_s']} |")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PYEOF
}

on_error_trap() {
    local exit_code=$?
    local line_no="${BASH_LINENO[0]:-unknown}"
    local cmd="${BASH_COMMAND:-unknown}"
    timing_record "${ACTIVE_QUERY}" "script_error" "trap" "0" "failed" "{\"line\": \"${line_no}\", \"command\": \"${cmd}\"}" || true
    log "    ✗ script aborted (code=${exit_code})"
    log "    Location: line=${line_no}"
    log "    Command: ${cmd}"
    if [[ -n "${Q_LOG:-}" ]] && [[ -f "${Q_LOG}" ]]; then
        log "    --- ${Q_LOG} (tail -n 60) ---"
        tail -n 60 "${Q_LOG}" | tee -a "${LOGFILE}" >/dev/null
        log "    --- end tail ---"
    fi
    exit "${exit_code}"
}

trap on_error_trap ERR

log_path() {
    local label="$1" path="$2" status="${3:-}"
    local flag=""
    [[ -e "${path}" ]] && flag="✓ exists" || flag="· pending"
    [[ -n "${status}" ]] && flag="${status}"
    log "    ${label}: ${path}  [${flag}]"
}

step_start() {
    log "━━━ STEP $1: $2 ━━━"
    STEP_T0="${SECONDS}"
}

step_done() {
    local elapsed=$(( SECONDS - STEP_T0 ))
    log "    ✓ done in ${elapsed}s"
}

log_failure_context_and_exit() {
    local stage_name="$1"
    local hint_path="$2"
    local qlog_path="${3:-}"
    local failed_cmd="${4:-${BASH_COMMAND:-unknown}}"
    local failed_line="${5:-${LINENO:-unknown}}"
    log "    ✗ ${stage_name} failed: missing ${hint_path}"
    log "    Location: line=${failed_line}"
    log "    Command: ${failed_cmd}"
    if [[ -n "${qlog_path}" ]] && [[ -f "${qlog_path}" ]]; then
        log "    --- ${qlog_path} (tail -n 40) ---"
        tail -n 40 "${qlog_path}" | tee -a "${LOGFILE}" >/dev/null
        log "    --- end tail ---"
    fi
    exit 1
}

require_file_or_exit() {
    local file_path="$1"
    local stage_name="$2"
    local qlog_path="${3:-}"
    if [[ ! -f "${file_path}" ]]; then
        log_failure_context_and_exit "${stage_name}" "file ${file_path}" "${qlog_path}"
    fi
}

require_dir_or_exit() {
    local dir_path="$1"
    local stage_name="$2"
    local qlog_path="${3:-}"
    if [[ ! -d "${dir_path}" ]]; then
        log_failure_context_and_exit "${stage_name}" "dir ${dir_path}" "${qlog_path}"
    fi
}

# --- Preconditions ---
if [[ ! -d "${RUN_DIR}/point_cloud" ]]; then
    echo "ERROR: Gaussian reconstruction missing at ${RUN_DIR}/point_cloud"
    exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "ERROR: dataset directory missing: ${DATASET_DIR}"
    exit 1
fi

# --- Step 0: build per-scene benchmark JSON ---
BENCHMARK_JSON_PATH="${LOG_ROOT}/${SCENE}_benchmark.json"
echo "=== Step 0: build benchmark JSON for ${SCENE} ==="
gs_python "${SCRIPT_DIR}/build_benchmark_json.py" \
    --benchmark-dir "${GS_ROOT}/benchmark" \
    --scenes "${SCENE}" \
    --output "${BENCHMARK_JSON_PATH}"

# --- Initialize entitybank ---
if [[ ! -f "${RUN_DIR}/entitybank/trajectory_samples.npz" ]]; then
    echo "[init] initialize entitybank: ${RUN_DIR}"
    conda run --no-capture-output \
        -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
        python "${SCRIPT_DIR}/export_entitybank.py" \
        --run-dir "${RUN_DIR}" \
        --max-entities 30 \
        --min-gaussians-per-entity 32
fi

log "======================================================="
log " 4DReSplat query pipeline — ${SCENE} benchmark"
log " RUN_DIR    : ${RUN_DIR}"
log " DATASET_DIR: ${DATASET_DIR}"
log " GPU        : ${GPU}"
log " QWEN_BATCH_SIZE: ${QWEN_BATCH_SIZE}"
log " LOGFILE    : ${LOGFILE}"
log " TIMING_JSONL: ${TIMING_JSONL}"
log "======================================================="
log ""

# ============================================================
# run_one_query_pre <query_text> <query_name>
# Serial stages: 3b+→3c→3c+→3d (GPU-heavy, not parallelized)
# ============================================================
run_one_query_pre() {
    local QUERY_TEXT="$1"
    local QUERY_NAME="$2"
    ACTIVE_QUERY="${QUERY_NAME}"

    local OUTPUT_ROOT="${RUN_DIR}/entitybank/query_guided/${QUERY_NAME}"
    local BASE_PLAN_PATH="${OUTPUT_ROOT}/query_plan.json"
    local PLAN_PATH="${BASE_PLAN_PATH}"
    local TRACKS_PATH="${OUTPUT_ROOT}/sam3_tracks/sam3_query_tracks.json"
    local REFINED_PLAN_PATH="${OUTPUT_ROOT}/query_plan_refined.json"
    local PROPOSAL_DIR="${OUTPUT_ROOT}/proposal_dir"
    local QUERY_ENTITYBANK_DIR="${OUTPUT_ROOT}/query_entitybank"
    local QUERY_RUN_DIR="${OUTPUT_ROOT}/query_worldtube_run"
    local QWEN_SELECTION="${QUERY_RUN_DIR}/entitybank/selected_query_qwen.json"
    local FINAL_VALIDATION="${OUTPUT_ROOT}/final_query_render_sourcebg/validation.json"
    local Q_LOG="${LOG_ROOT}/${QUERY_NAME}.log"

    log ""
    log "▶▶▶ [pre] ${QUERY_NAME}: \"${QUERY_TEXT}\""
    mkdir -p "${OUTPUT_ROOT}"

    if [[ -f "${FINAL_VALIDATION}" ]]; then
        log "    [SKIP pre] validation.json already exists"
        timing_record "${QUERY_NAME}" "query_total" "already_done" "0" "skip"
        ACTIVE_QUERY="__pipeline__"
        return 0
    fi

    # --- 3b fallback ---
    if [[ ! -f "${TRACKS_PATH}" ]]; then
        log "    [SKIP 3b] tracks missing; writing empty tracks"
        python3 -c "
import json; from pathlib import Path
p = Path('${BASE_PLAN_PATH}'); qp = json.loads(p.read_text())
tp = Path('${TRACKS_PATH}'); tp.parent.mkdir(parents=True, exist_ok=True)
payload = {'schema_version':2,'query':qp.get('query',''),'subjects':qp.get('subjects',[]),
  'sam_backend':'empty_fallback',
  'sam_time_window':{'start_frame':None,'end_frame':None,'confidence':'no_detection','active_frame_count':0},
  'tracks':[],'active_frame_indices':[],'negative_query':False}
tp.write_text(json.dumps(payload,indent=2,ensure_ascii=False))
"
        timing_record "${QUERY_NAME}" "3b" "sam3_infer" "0" "skip_batch_only"
    else
        timing_record "${QUERY_NAME}" "3b" "sam3_infer" "0" "skip"
    fi

    # --- 3b+: refine temporal window ---
    if [[ ! -f "${REFINED_PLAN_PATH}" ]]; then
        step_start "3b+" "Refine temporal window (SAM active-frame range)"
        local T_STEP="${SECONDS}"
        conda run --no-capture-output \
            -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
            python3 -c "
import json, sys
from pathlib import Path
sys.path.insert(0, '${GS_ROOT}')
from scripts.query_with_entity_cache import _read_json, _write_json, _refine_query_plan_time_window
refined = _refine_query_plan_time_window(
    query_plan_path=Path('${PLAN_PATH}'),
    tracks_path=Path('${TRACKS_PATH}'),
    output_dir=Path('${OUTPUT_ROOT}'),
)
print('refined plan:', refined)
" 2>&1 | tee -a "${Q_LOG}" "${LOGFILE}"
        timing_record "${QUERY_NAME}" "3b+" "refine_window" "$(( SECONDS - T_STEP ))" "done"
        step_done
        timing_record "${QUERY_NAME}" "3b+" "refined_plan_probe" "0" "done" "$(probe_path_json "${REFINED_PLAN_PATH}")"
    else
        log "    [SKIP 3b+] refined plan already exists"
        timing_record "${QUERY_NAME}" "3b+" "refine_window" "0" "skip"
    fi
    [[ -f "${REFINED_PLAN_PATH}" ]] && PLAN_PATH="${REFINED_PLAN_PATH}"

    # --- Negative-query short circuit ---
    local IS_NEGATIVE
    IS_NEGATIVE=$(python3 -c "
import json; from pathlib import Path
p = Path('${BASE_PLAN_PATH}')
if not p.exists(): print('false')
else:
    d = json.loads(p.read_text())
    s = d.get('subjects', None)
    print('true' if d.get('negative_query', False) or (isinstance(s, list) and len(s)==0) else 'false')
" 2>/dev/null || echo "false")

    if [[ "${IS_NEGATIVE}" == "true" ]]; then
        log "    [negative] subjects=[] → empty selection; skip 3c/3c+/3d/3e"
        mkdir -p "${QUERY_RUN_DIR}/entitybank"
        python3 -c "
import json; from pathlib import Path
p = Path('${QWEN_SELECTION}'); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({'empty':True,'selected':[],'reason':'negative_query_kb'},indent=2,ensure_ascii=False))
"
        timing_record "${QUERY_NAME}" "3c" "build_proposal_dir" "0" "skip_negative"
        timing_record "${QUERY_NAME}" "3c+" "export_query_entitybank" "0" "skip_negative"
        timing_record "${QUERY_NAME}" "3d" "select_entities" "0" "skip_negative"
        timing_record "${QUERY_NAME}" "query_total" "query_total" "0" "done"
        log "◀◀◀ [pre done] ${QUERY_NAME} (negative)"
        ACTIVE_QUERY="__pipeline__"
        log ""
        return 0
    fi

    # --- 3c: Gaussian entity extraction ---
    if [[ ! -d "${PROPOSAL_DIR}" ]] || [[ ! -f "${PROPOSAL_DIR}/entities.json" ]]; then
        step_start "3c" "Gaussian entity extraction (build_query_proposal_dir)"
        local T_STEP="${SECONDS}"
        conda run --no-capture-output \
            -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
            python "${SCRIPT_DIR}/build_query_proposal_dir.py" \
            --run-dir "${RUN_DIR}" \
            --dataset-dir "${DATASET_DIR}" \
            --tracks-path "${TRACKS_PATH}" \
            --output-dir "${PROPOSAL_DIR}" \
            --max-track-frames 16 \
            --proposal-keep-ratio 0.03 \
            --min-gaussians 256 \
            --max-gaussians 4096 \
            2>&1 | tee -a "${Q_LOG}" "${LOGFILE}"
        timing_record "${QUERY_NAME}" "3c" "build_proposal_dir" "$(( SECONDS - T_STEP ))" "done"
        require_file_or_exit "${PROPOSAL_DIR}/entities.json" "Step 3c" "${Q_LOG}"
        step_done
        timing_record "${QUERY_NAME}" "3c" "proposal_probe" "0" "done" "$(probe_path_json "${PROPOSAL_DIR}")"
    else
        log "    [SKIP 3c] proposal_dir already exists"
        timing_record "${QUERY_NAME}" "3c" "build_proposal_dir" "0" "skip"
    fi

    # --- 3c+: export query entitybank ---
    if [[ ! -d "${QUERY_ENTITYBANK_DIR}" ]] || [[ ! -f "${QUERY_ENTITYBANK_DIR}/entities.json" ]]; then
        step_start "3c+" "Export query entitybank"
        local T_STEP="${SECONDS}"
        conda run --no-capture-output \
            -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
            python "${SCRIPT_DIR}/export_entitybank.py" \
            --run-dir "${RUN_DIR}" \
            --proposal-dir "${PROPOSAL_DIR}" \
            --proposal-strict \
            --output-dir "${QUERY_ENTITYBANK_DIR}" \
            --max-entities 12 \
            --min-gaussians-per-entity 32 \
            2>&1 | tee -a "${Q_LOG}" "${LOGFILE}"
        timing_record "${QUERY_NAME}" "3c+" "export_query_entitybank" "$(( SECONDS - T_STEP ))" "done"
        require_file_or_exit "${QUERY_ENTITYBANK_DIR}/entities.json" "Step 3c+" "${Q_LOG}"
        step_done
        timing_record "${QUERY_NAME}" "3c+" "entitybank_probe" "0" "done" "$(probe_path_json "${QUERY_ENTITYBANK_DIR}")"
    else
        log "    [SKIP 3c+] query_entitybank already exists"
        timing_record "${QUERY_NAME}" "3c+" "export_query_entitybank" "0" "skip"
    fi

    # --- Symlinks ---
    mkdir -p "${QUERY_RUN_DIR}"
    [[ -f "${RUN_DIR}/config.yaml" ]] && ln -sfn "${RUN_DIR}/config.yaml" "${QUERY_RUN_DIR}/config.yaml"
    ln -sfn "${RUN_DIR}/point_cloud"  "${QUERY_RUN_DIR}/point_cloud"
    ln -sfn "${RUN_DIR}/test"         "${QUERY_RUN_DIR}/test"
    ln -sfn "${QUERY_ENTITYBANK_DIR}" "${QUERY_RUN_DIR}/entitybank"
    timing_record "${QUERY_NAME}" "link" "create_symlinks" "0" "done"

    # --- 3d: entity selection ---
    local ACTIVE_PLAN="${REFINED_PLAN_PATH}"
    [[ ! -f "${ACTIVE_PLAN}" ]] && ACTIVE_PLAN="${PLAN_PATH}"
    if [[ -f "${ACTIVE_PLAN}" ]] && [[ -f "${BASE_PLAN_PATH}" ]]; then
        local PLAN_COUNTS
        PLAN_COUNTS=$(python3 - <<PYEOF
import json
from pathlib import Path
def n_sub(p):
    try: d=json.loads(Path(p).read_text()); return len([s for s in (d.get("query_subject_phrases") or []) if str(s).strip()])
    except: return 0
print(n_sub("${BASE_PLAN_PATH}"), n_sub("${ACTIVE_PLAN}"))
PYEOF
)
        local BASE_N ACTIVE_N
        BASE_N="$(echo "${PLAN_COUNTS}" | awk '{print $1}')"
        ACTIVE_N="$(echo "${PLAN_COUNTS}" | awk '{print $2}')"
        if [[ "${BASE_N}" -gt 0 ]] && [[ "${ACTIVE_N}" -eq 0 ]]; then
            log "    [WARN] refined plan missing query_subject_phrases; 3d falls back to query_plan.json"
            ACTIVE_PLAN="${BASE_PLAN_PATH}"
        fi
    fi
    if [[ ! -f "${QWEN_SELECTION}" ]]; then
        step_start "3d" "Entity selection"
        local T_STEP="${SECONDS}"
        set +eo pipefail
        conda run --no-capture-output \
            -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
            python "${SCRIPT_DIR}/select_qwen_query_entities.py" \
            --query "${QUERY_TEXT}" \
            --query-plan-path "${ACTIVE_PLAN}" \
            --output-path "${QWEN_SELECTION}" \
            --assignments-path "${QUERY_RUN_DIR}/entitybank/entities.json" \
            2>&1 | tee -a "${Q_LOG}" "${LOGFILE}"
        _SEL_EC="${PIPESTATUS[0]}"
        set -eo pipefail
        if [[ "${_SEL_EC}" -ne 0 ]]; then
            log "    ⚠ Step 3d failed (exit=${_SEL_EC}); empty selection → skip 3e"
            python3 -c "
import json; from pathlib import Path
p = Path('${QWEN_SELECTION}'); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({'empty':True,'selected':[],'reason':'selection_error'},indent=2,ensure_ascii=False))
"
            timing_record "${QUERY_NAME}" "3d" "select_entities" "$(( SECONDS - T_STEP ))" "error_skipped"
        else
            timing_record "${QUERY_NAME}" "3d" "select_entities" "$(( SECONDS - T_STEP ))" "done"
            require_file_or_exit "${QWEN_SELECTION}" "Step 3d" "${Q_LOG}"
            timing_record "${QUERY_NAME}" "3d" "selection_probe" "0" "done" "$(probe_path_json "${QWEN_SELECTION}")"
        fi
        step_done
    else
        log "    [SKIP 3d] selection already exists"
        timing_record "${QUERY_NAME}" "3d" "select_entities" "0" "skip"
    fi

    log "◀◀◀ [pre done] ${QUERY_NAME}"
    ACTIVE_QUERY="__pipeline__"
    log ""
}

# ============================================================
# run_one_query_render <query_text> <query_name>
# Render stage: 3e (parallel child shells)
# ============================================================
run_one_query_render() {
    local QUERY_TEXT="$1"
    local QUERY_NAME="$2"
    local Q_START="${SECONDS}"

    local OUTPUT_ROOT="${RUN_DIR}/entitybank/query_guided/${QUERY_NAME}"
    local QUERY_RUN_DIR="${OUTPUT_ROOT}/query_worldtube_run"
    local QWEN_SELECTION="${QUERY_RUN_DIR}/entitybank/selected_query_qwen.json"
    local FINAL_RENDER_DIR="${OUTPUT_ROOT}/final_query_render_sourcebg"
    local FINAL_VALIDATION="${FINAL_RENDER_DIR}/validation.json"
    local Q_LOG="${LOG_ROOT}/${QUERY_NAME}.log"

    if [[ -f "${FINAL_VALIDATION}" ]]; then
        log "    [SKIP 3e] ${QUERY_NAME}: validation.json already exists"
        timing_record "${QUERY_NAME}" "query_total" "already_done" "0" "skip"
        return 0
    fi

    # Skip when pre-stage incomplete (no selection)
    if [[ ! -f "${QWEN_SELECTION}" ]]; then
        log "    [SKIP 3e] ${QUERY_NAME}: selection missing (pre incomplete)"
        return 0
    fi

    # Skip negative queries with empty selection
    local IS_EMPTY
    IS_EMPTY=$(python3 -c "
import json; from pathlib import Path
d = json.loads(Path('${QWEN_SELECTION}').read_text())
print('true' if d.get('empty', False) else 'false')
" 2>/dev/null || echo "false")
    if [[ "${IS_EMPTY}" == "true" ]]; then
        log "    [SKIP 3e] ${QUERY_NAME}: negative/empty selection"
        return 0
    fi

    if [[ "${RUN_SKIP_3E:-0}" == "1" || "${RUN_SKIP_3E:-0}" == "true" || "${RUN_SKIP_3E:-0}" == "yes" ]]; then
        log "    [SKIP 3e] ${QUERY_NAME}: RUN_SKIP_3E is set"
        timing_record "${QUERY_NAME}" "3e" "render_query_video" "0" "skipped" "{\"reason\":\"RUN_SKIP_3E\"}"
        timing_record "${QUERY_NAME}" "query_total" "query_total" "0" "done"
        return 0
    fi

    log "  [3e] rendering ${QUERY_NAME}"
    local T_STEP="${SECONDS}"
    conda run --no-capture-output \
        -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
        python "${SCRIPT_DIR}/render_query_video.py" \
        --run-dir "${QUERY_RUN_DIR}" \
        --dataset-dir "${DATASET_DIR}" \
        --selection-path "${QWEN_SELECTION}" \
        --output-dir "${FINAL_RENDER_DIR}" \
        --background-mode source \
        --fps 6 \
        --stride 1 \
        --no-overlay \
        2>&1 | tee -a "${Q_LOG}" "${LOGFILE}"
    timing_record "${QUERY_NAME}" "3e" "render_query_video" "$(( SECONDS - T_STEP ))" "done"
    if [[ ! -f "${FINAL_VALIDATION}" ]]; then
        log "    ✗ [3e] ${QUERY_NAME}: validation.json missing"
        return 1
    fi
    local Q_ELAPSED=$(( SECONDS - Q_START ))
    log "  ✓ [3e] finished ${QUERY_NAME} in ${Q_ELAPSED}s"
    timing_record "${QUERY_NAME}" "query_total" "query_total" "${Q_ELAPSED}" "done"
}

# --- Read all queries from benchmark JSON ---
mapfile -t ALL_QUERY_IDS < <(python3 -c "
import json
data = json.load(open('${BENCHMARK_JSON_PATH}'))
qs = data.get('queries', data) if isinstance(data, dict) else data
if isinstance(qs, dict): qs = list(qs.values())
for q in qs:
    print(q['query_id'])
" 2>/dev/null)

mapfile -t ALL_QUERY_TEXTS < <(python3 -c "
import json
data = json.load(open('${BENCHMARK_JSON_PATH}'))
qs = data.get('queries', data) if isinstance(data, dict) else data
if isinstance(qs, dict): qs = list(qs.values())
for q in qs:
    print(q.get('question') or q.get('query_text', ''))
" 2>/dev/null)

if [[ -n "${RUN_QUERY_IDS:-}" || -n "${RUN_QUERY_LIMIT:-}" ]]; then
    FILTERED_JSON="$(python3 - "${RUN_QUERY_IDS:-}" "${RUN_QUERY_LIMIT:-}" "${ALL_QUERY_IDS[@]}" <<'PYEOF'
import json
import sys

query_ids_raw = sys.argv[1].strip()
limit_raw = sys.argv[2].strip()
ids = sys.argv[3:]
keep = None
if query_ids_raw:
    keep = {item.strip() for item in query_ids_raw.replace(",", " ").split() if item.strip()}
selected = []
for idx, qid in enumerate(ids):
    if keep is not None and qid not in keep:
        continue
    selected.append(idx)
if limit_raw:
    selected = selected[: max(int(limit_raw), 0)]
print(json.dumps(selected))
PYEOF
)"
    mapfile -t ALL_QUERY_IDS < <(python3 - "${FILTERED_JSON}" "${ALL_QUERY_IDS[@]}" <<'PYEOF'
import json
import sys
selected = json.loads(sys.argv[1])
values = sys.argv[2:]
for idx in selected:
    print(values[int(idx)])
PYEOF
)
    mapfile -t ALL_QUERY_TEXTS < <(python3 - "${FILTERED_JSON}" "${ALL_QUERY_TEXTS[@]}" <<'PYEOF'
import json
import sys
selected = json.loads(sys.argv[1])
values = sys.argv[2:]
for idx in selected:
    print(values[int(idx)])
PYEOF
)
fi

N_QUERIES="${#ALL_QUERY_IDS[@]}"
log "Loaded ${N_QUERIES} queries from ${BENCHMARK_JSON_PATH}"
if [[ -n "${RUN_QUERY_IDS:-}" || -n "${RUN_QUERY_LIMIT:-}" ]]; then
    log "Query subset: RUN_QUERY_IDS=${RUN_QUERY_IDS:-<unset>} RUN_QUERY_LIMIT=${RUN_QUERY_LIMIT:-<unset>}"
fi

# --- Batch Step 3a: Qwen plans all queries in one shot ---
PIPELINE_T0="${SECONDS}"

log "======================================================="
log " Step 3a batch Qwen query plan (${N_QUERIES} queries)"
log "======================================================="

BATCH_PLAN_T0="${SECONDS}"
MANIFEST_PATH="/tmp/${SCENE}_plan_manifest_${TIMESTAMP}.json"
FILTERED_QUERY_IDS_FILE="/tmp/${SCENE}_query_ids_${TIMESTAMP}.txt"
FILTERED_QUERY_TEXTS_FILE="/tmp/${SCENE}_query_texts_${TIMESTAMP}.txt"
printf '%s\n' "${ALL_QUERY_IDS[@]}" > "${FILTERED_QUERY_IDS_FILE}"
printf '%s\n' "${ALL_QUERY_TEXTS[@]}" > "${FILTERED_QUERY_TEXTS_FILE}"

python3 - <<PYEOF
import json, os
from pathlib import Path

run_dir = "${RUN_DIR}"
dataset_dir = "${DATASET_DIR}"
manifest_path = "${MANIFEST_PATH}"
query_ids_path = "${FILTERED_QUERY_IDS_FILE}"
query_texts_path = "${FILTERED_QUERY_TEXTS_FILE}"

query_ids = Path(query_ids_path).read_text(encoding="utf-8").splitlines()
query_texts = Path(query_texts_path).read_text(encoding="utf-8").splitlines()
if len(query_ids) != len(query_texts):
    raise RuntimeError(f"query id/text count mismatch: {len(query_ids)} != {len(query_texts)}")

items = []
for qid, qtext in zip(query_ids, query_texts):
    out_dir  = Path(run_dir) / "entitybank" / "query_guided" / qid
    out_path = out_dir / "query_plan.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    items.append({
        "query_id":   qid,
        "query_text": qtext,
        "dataset_dir": dataset_dir,
        "output_path": str(out_path),
    })

Path(manifest_path).write_text(json.dumps(items, indent=2, ensure_ascii=False))
print(f"Manifest written: {len(items)} queries → {manifest_path}")
PYEOF

PENDING=$(python3 -c "
import json
items = json.load(open('${MANIFEST_PATH}'))
print(sum(1 for i in items if not __import__('pathlib').Path(i['output_path']).exists()))
")

if [[ "${PENDING}" -gt 0 ]]; then
    step_start "3a" "Batch Qwen query plan (single load; ${PENDING}/${N_QUERIES} pending)"
    conda run --no-capture-output \
        -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
        python "${SCRIPT_DIR}/batch_plan_queries.py" \
        --manifest "${MANIFEST_PATH}" \
        --coarse-stride "${GSAM2_FRAME_SUBSAMPLE_STRIDE}" \
        --batch-size "${QWEN_BATCH_SIZE}" \
        2>&1 | tee -a "${LOG_ROOT}/batch_plan.log" "${LOGFILE}"
    step_done
else
    log "    [SKIP 3a] all ${N_QUERIES} query_plan.json files exist"
fi
log "    Step 3a batch finished in $(( SECONDS - BATCH_PLAN_T0 ))s"
log ""

# --- Batch Step 3b: SAM3.1 text tracking (single load) ---
log "======================================================="
log " Step 3b batch SAM3.1 tracking (${N_QUERIES} queries)"
log "======================================================="

SAM_MANIFEST_PATH="/tmp/${SCENE}_sam_manifest_${TIMESTAMP}.json"
python3 - <<PYEOF
import json
from pathlib import Path

run_dir    = "${RUN_DIR}"
dataset_dir = "${DATASET_DIR}"
manifest_in = "${MANIFEST_PATH}"
manifest_out = "${SAM_MANIFEST_PATH}"

items = json.loads(Path(manifest_in).read_text())
sam_items = []
for item in items:
    qid = item["query_id"]
    plan_path = item["output_path"]
    output_dir = str(Path(run_dir) / "entitybank" / "query_guided" / qid / "sam3_tracks")
    sam_items.append({
        "query_id":        qid,
        "dataset_dir":     dataset_dir,
        "query_plan_path": plan_path,
        "output_dir":      output_dir,
    })
Path(manifest_out).write_text(json.dumps(sam_items, indent=2, ensure_ascii=False))
print(f"SAM manifest: {len(sam_items)} queries → {manifest_out}")
PYEOF

SAM_PENDING=$(python3 -c "
import json
from pathlib import Path
items = json.load(open('${SAM_MANIFEST_PATH}'))
print(sum(1 for i in items if not (Path(i['output_dir']) / 'sam3_query_tracks.json').exists()))
")

BATCH_SAM_T0="${SECONDS}"
if [[ "${SAM_PENDING}" -gt 0 ]]; then
    step_start "3b" "Batch SAM3.1 text tracking (${SAM_PENDING}/${N_QUERIES} pending)"
    SAM3_NUM_PROCS="${SAM3_NUM_PROCS:-1}"
    SAM3_FRAME_STRIDE="${SAM3_FRAME_STRIDE:-20}"
    SAM3_PRELOAD_FRAMES="${SAM3_PRELOAD_FRAMES:-none}"
    SAM3_KEEP_VIDEO_ON_GPU="${SAM3_KEEP_VIDEO_ON_GPU:-0}"
    if [[ "${SAM3_FRAME_STRIDE}" -le 4 ]] && [[ "${SAM3_KEEP_VIDEO_ON_GPU}" == "1" || "${SAM3_KEEP_VIDEO_ON_GPU}" == "true" || "${SAM3_KEEP_VIDEO_ON_GPU}" == "yes" ]]; then
        log "    [3b] SAM3_FRAME_STRIDE=${SAM3_FRAME_STRIDE}: VRAM pressure high; forcing keep-video-on-gpu off to avoid OOM"
        SAM3_KEEP_VIDEO_ON_GPU=0
    fi
    declare -a SAM3_IO_ARGS=()
    if [[ "${SAM3_PRELOAD_FRAMES}" != "none" ]]; then
        SAM3_IO_ARGS+=(--preload-frames "${SAM3_PRELOAD_FRAMES}")
    fi
    if [[ "${SAM3_KEEP_VIDEO_ON_GPU}" == "1" || "${SAM3_KEEP_VIDEO_ON_GPU}" == "true" || "${SAM3_KEEP_VIDEO_ON_GPU}" == "yes" ]]; then
        SAM3_IO_ARGS+=(--keep-video-on-gpu)
    fi
    log "    [3b] IO: SAM3_FRAME_STRIDE=${SAM3_FRAME_STRIDE}, SAM3_PRELOAD_FRAMES=${SAM3_PRELOAD_FRAMES}, SAM3_KEEP_VIDEO_ON_GPU=${SAM3_KEEP_VIDEO_ON_GPU}"
    SAM3_SESSION_RESET_INTERVAL="${SAM3_SESSION_RESET_INTERVAL:-1}"
    if [[ "${SAM3_NUM_PROCS}" -le 1 ]]; then
        set +eo pipefail
        conda run --no-capture-output \
            -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
            python "${SCRIPT_DIR}/batch_run_sam3_text.py" \
            --manifest "${SAM_MANIFEST_PATH}" \
            --sam3-checkpoint "${GS_ROOT}/sam3.1/sam3.1_multiplex.pt" \
            --frame-stride "${SAM3_FRAME_STRIDE}" \
            --session-reset-interval "${SAM3_SESSION_RESET_INTERVAL}" \
            "${SAM3_IO_ARGS[@]}" \
            2>&1 | tee -a "${LOG_ROOT}/batch_sam.log" "${LOGFILE}"
        _SAM_EC="${PIPESTATUS[0]}"
        set -eo pipefail
        if [[ "${_SAM_EC}" -ne 0 ]]; then
            _SAM_MISSING=$(python3 - "${SAM_MANIFEST_PATH}" <<'PYEOF'
import json, sys
from pathlib import Path
items = json.load(open(sys.argv[1]))
print(sum(1 for i in items if not (Path(i['output_dir']) / 'sam3_query_tracks.json').exists()))
PYEOF
)
            if [[ "${_SAM_EC}" -eq 134 ]] && [[ "${_SAM_MISSING}" -eq 0 ]]; then
                log "    ⚠ SAM3 exited 134 (SIGABRT during teardown); tracks written — continuing"
            else
                log "    ✗ SAM3 failed (exit=${_SAM_EC}, missing ${_SAM_MISSING} tracks)"
                exit 1
            fi
        fi
    else
        log "    [3b] parallel mode: SAM3_NUM_PROCS=${SAM3_NUM_PROCS}"
        SAM_SHARD_PREFIX="/tmp/${SCENE}_sam_manifest_${TIMESTAMP}_shard"
        python3 - "${SAM_MANIFEST_PATH}" "${SAM_SHARD_PREFIX}" "${SAM3_NUM_PROCS}" <<'PYEOF'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
prefix = sys.argv[2]
num_procs = max(int(sys.argv[3]), 1)
items = json.loads(manifest_path.read_text())
shards = [[] for _ in range(num_procs)]
for idx, item in enumerate(items):
    shards[idx % num_procs].append(item)
for i, shard in enumerate(shards):
    out = Path(f"{prefix}_{i}.json")
    out.write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"shard[{i}]={len(shard)} -> {out}")
PYEOF

        declare -a SAM_PIDS=()
        declare -a SAM_SHARD_IDS=()
        for (( shard=0; shard<SAM3_NUM_PROCS; shard++ )); do
            SHARD_MANIFEST="${SAM_SHARD_PREFIX}_${shard}.json"
            SHARD_LOG="${LOG_ROOT}/batch_sam_p${shard}.log"
            log "    [3b] start worker ${shard}: ${SHARD_MANIFEST}"
            (
                conda run --no-capture-output \
                    -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
                    python "${SCRIPT_DIR}/batch_run_sam3_text.py" \
                    --manifest "${SHARD_MANIFEST}" \
                    --sam3-checkpoint "${GS_ROOT}/sam3.1/sam3.1_multiplex.pt" \
                    --frame-stride "${SAM3_FRAME_STRIDE}" \
                    "${SAM3_IO_ARGS[@]}" \
                    2>&1 | tee -a "${SHARD_LOG}" "${LOGFILE}"
            ) &
            SAM_PIDS+=("$!")
            SAM_SHARD_IDS+=("${shard}")
        done

        SAM_FAILED=0
        for (( i=0; i<${#SAM_PIDS[@]}; i++ )); do
            pid="${SAM_PIDS[$i]}"
            shard="${SAM_SHARD_IDS[$i]}"
            if wait "${pid}"; then
                log "    [3b] worker ${shard} done"
            else
                _SHARD_EC=$?
                _SHARD_MANIFEST="${SAM_SHARD_PREFIX}_${shard}.json"
                _SHARD_MISSING=$(python3 - "${_SHARD_MANIFEST}" <<'PYEOF'
import json, sys
from pathlib import Path
items = json.load(open(sys.argv[1]))
print(sum(1 for i in items if not (Path(i['output_dir']) / 'sam3_query_tracks.json').exists()))
PYEOF
)
                if [[ "${_SHARD_EC}" -eq 134 ]] && [[ "${_SHARD_MISSING}" -eq 0 ]]; then
                    log "    ⚠ [3b] worker ${shard} exited 134 (SIGABRT during teardown); tracks written — continuing"
                else
                    log "    [3b] worker ${shard} failed (pid=${pid}, exit=${_SHARD_EC}, missing=${_SHARD_MISSING})"
                    SAM_FAILED=1
                fi
            fi
        done
        if [[ "${SAM_FAILED}" -ne 0 ]]; then
            log "    ✗ Step 3b parallel run failed"
            exit 1
        fi
    fi
    step_done
else
    log "    [SKIP 3b] all ${N_QUERIES} sam3_query_tracks.json exist"
fi
log "    Step 3b batch finished in $(( SECONDS - BATCH_SAM_T0 ))s"
log ""

# --- After batch 3b: write empty tracks for missing outputs ---
log " [3b-fallback] writing empty tracks for queries missing sam3_query_tracks.json …"
python3 - "${RUN_DIR}" "${ALL_QUERY_IDS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
query_ids = sys.argv[2:]
eb = run_dir / "entitybank" / "query_guided"
written = []
for qid in query_ids:
    qdir = eb / qid
    tracks_path = qdir / "sam3_tracks" / "sam3_query_tracks.json"
    if tracks_path.exists():
        continue
    plan_path = qdir / "query_plan.json"
    if not plan_path.exists():
        continue
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    qp = json.loads(plan_path.read_text())
    payload = {
        "schema_version": 2,
        "query": qp.get("query", ""),
        "subjects": qp.get("subjects", []),
        "sam_backend": "empty_fallback",
        "sam_time_window": {"start_frame": None, "end_frame": None, "confidence": "no_detection", "active_frame_count": 0},
        "tracks": [],
        "active_frame_indices": [],
        "negative_query": False,
    }
    tracks_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    written.append(qid)
if written:
    print(f"[3b-fallback] wrote empty tracks for: {written}")
else:
    print("[3b-fallback] all queries have tracks.json, nothing to do")
PYEOF
log ""

# --- quick-eval paths ---
QUICK_EVAL_REPORT_DIR="${GS_ROOT}/reports/${SCENE}_eval"
QUICK_EVAL_BENCHMARK_JSON="${QUICK_EVAL_REPORT_DIR}/${SCENE}_benchmark.json"
QUICK_EVAL_QUERY_ROOT_MAP="${QUICK_EVAL_REPORT_DIR}/query_root_map.json"
QUICK_EVAL_DATASET_DIR_MAP="${QUICK_EVAL_REPORT_DIR}/dataset_dir_map.json"
QUICK_EVAL_JSON="${QUICK_EVAL_REPORT_DIR}/eval_results.json"
QUICK_EVAL_MD="${QUICK_EVAL_REPORT_DIR}/eval_results.md"

run_quick_eval() {
    log "    [quick-eval] running quick evaluation …"
    local T_EVAL="${SECONDS}"
    mkdir -p "${QUICK_EVAL_REPORT_DIR}"
    python3 - "${RUN_DIR}" "${DATASET_DIR}" "${BENCHMARK_JSON_PATH}" \
              "${QUICK_EVAL_BENCHMARK_JSON}" "${QUICK_EVAL_QUERY_ROOT_MAP}" "${QUICK_EVAL_DATASET_DIR_MAP}" \
    <<'PYEOF'
import json, sys
from pathlib import Path

run_dir      = Path(sys.argv[1])
dataset_dir  = Path(sys.argv[2])
bm_src       = Path(sys.argv[3])
bm_dst       = Path(sys.argv[4])
root_map_dst = Path(sys.argv[5])
ds_map_dst   = Path(sys.argv[6])

data = json.loads(bm_src.read_text())
qs = data.get("queries", data) if isinstance(data, dict) else data
if isinstance(qs, dict): qs = list(qs.values())

root_map, ds_map = {}, {}
for q in qs:
    qid = str(q.get("query_id", q.get("id")))
    if not qid or qid == "None": continue
    root_map[qid] = str(run_dir / "entitybank" / "query_guided" / qid)
    ds_map[qid]   = str(dataset_dir)

bm_dst.write_text(json.dumps(qs, indent=2, ensure_ascii=False))
root_map_dst.write_text(json.dumps(root_map, indent=2))
ds_map_dst.write_text(json.dumps(ds_map, indent=2))
ready = sum(1 for p in root_map.values()
            if (Path(p) / "final_query_render_sourcebg" / "validation.json").exists()
            or (Path(p) / "sam3_tracks" / "sam3_query_tracks.json").exists())
print(f"quick-eval: {ready}/{len(root_map)} queries ready")
PYEOF

    conda run --no-capture-output \
        -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
        python "${SCRIPT_DIR}/evaluate_ours_benchmark.py" \
        --benchmark "${QUICK_EVAL_BENCHMARK_JSON}" \
        --benchmark-root "${GS_ROOT}/benchmark" \
        --query-root-map "${QUICK_EVAL_QUERY_ROOT_MAP}" \
        --dataset-dir-map "${QUICK_EVAL_DATASET_DIR_MAP}" \
        --output-json "${QUICK_EVAL_JSON}" \
        --output-md "${QUICK_EVAL_MD}" \
        --skip-missing \
        2>&1 | tee -a "${LOGFILE}" >/dev/null
    python3 - "${QUICK_EVAL_JSON}" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists(): print("  [quick-eval] eval_results.json missing"); sys.exit(0)
d = json.load(open(p))
s = d.get("summary", {})
print(f"  [quick-eval] valid={s.get('valid_queries',0)}  Acc={s.get('Acc',0)*100:.1f}%  tIoU={s.get('tIoU',0)*100:.2f}%  tPrec={s.get('tPrec',0)*100:.2f}%  tRec={s.get('tRec',0)*100:.2f}%")
PYEOF
    log "    [quick-eval] finished in $((SECONDS - T_EVAL))s  output: ${QUICK_EVAL_MD}"
}

# --- Pass 1: serial preprocess 3b+→3c→3c+→3d ---
log "======================================================="
log " Pass 1/2: preprocess (3b+/3c/3c+/3d) — serial ${N_QUERIES} queries"
log "======================================================="
for (( i=0; i<N_QUERIES; i++ )); do
    run_one_query_pre "${ALL_QUERY_TEXTS[$i]}" "${ALL_QUERY_IDS[$i]}"
done
log " Pass 1 finished in $(( SECONDS - PIPELINE_T0 ))s"
log ""

# --- Pass 2: parallel render 3e ---
SCENE_RENDER_JOBS="${SCENE_RENDER_JOBS:-4}"
log "======================================================="
log " Pass 2/2: render stage (3e) — jobs=${SCENE_RENDER_JOBS} ${N_QUERIES} queries"
log "======================================================="
RENDER_T0="${SECONDS}"

if [[ "${SCENE_RENDER_JOBS}" -le 1 ]]; then
    for (( i=0; i<N_QUERIES; i++ )); do
        run_one_query_render "${ALL_QUERY_TEXTS[$i]}" "${ALL_QUERY_IDS[$i]}"
    done
else
    render_fail=0
    active_jobs=0
    for (( i=0; i<N_QUERIES; i++ )); do
        (
            run_one_query_render "${ALL_QUERY_TEXTS[$i]}" "${ALL_QUERY_IDS[$i]}"
        ) &
        active_jobs=$(( active_jobs + 1 ))
        if [[ "${active_jobs}" -ge "${SCENE_RENDER_JOBS}" ]]; then
            if ! wait -n; then render_fail=1; fi
            active_jobs=$(( active_jobs - 1 ))
        fi
    done
    while [[ "${active_jobs}" -gt 0 ]]; do
        if ! wait -n; then render_fail=1; fi
        active_jobs=$(( active_jobs - 1 ))
    done
    if [[ "${render_fail}" -ne 0 ]]; then
        log "  [3e] warning: some renders failed; continuing pipeline"
    fi
fi
log " Pass 2 render finished in $(( SECONDS - RENDER_T0 ))s"
log ""

run_quick_eval || log "    [quick-eval] evaluation failed; continuing"

TOTAL=$(( SECONDS - PIPELINE_T0 ))
timing_record "__pipeline__" "pipeline_total" "pipeline_total" "${TOTAL}" "done"
timing_summarize
log "======================================================="
log " All done in ${TOTAL}s ($(( TOTAL/60 ))m$(( TOTAL%60 ))s)"
log " Log: ${LOGFILE}"
log " Timing detail (JSONL): ${TIMING_JSONL}"
log " Timing summary (MD): ${TIMING_SUMMARY_MD}"
log " Timing summary (JSON): ${TIMING_SUMMARY_JSON}"
log " Next: bash scripts/eval_scene_benchmark.sh ${SCENE}"
log "======================================================="
