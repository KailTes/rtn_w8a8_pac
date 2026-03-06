#!/bin/bash
# ============================================================
# RTN W8A8 INT8 量化 + PPL 评测 (昇腾 NPU, omni-infer)
#
# 用法:
#   bash rtn_eval.sh quantize  /path/to/model  — RTN W8A8 离线量化 (CPU)
#   bash rtn_eval.sh serve     /path/to/model  — 启动 vllm serve (omni-infer)
#   bash rtn_eval.sh eval      /path/to/model  — 通过 API 评测 PPL
#   bash rtn_eval.sh stop                      — 停止 vllm serve
#   bash rtn_eval.sh eval_fp16 /path/to/model  — FP16 baseline (serve + eval)
#   bash rtn_eval.sh eval_w8a8 /path/to/model  — W8A8 评测 (serve + eval)
#   bash rtn_eval.sh all       /path/to/model  — 全部依次执行
#
# 环境变量 (可选):
#   API_BASE            — API 地址 (默认: http://7.150.11.4:8000)
#   SERVE_PORT          — vllm serve 端口 (默认: 8000)
#   TP_SIZE             — tensor parallel size (默认: 1)
#   RUN_PANGU_SCRIPT    — run_pangu.sh 路径 (默认: 自动查找)
#   EXTRA_SERVE_ARGS    — 额外的 vllm serve 参数
# ============================================================

set -uo pipefail

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVE_PORT="${SERVE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
API_BASE="${API_BASE:-http://7.150.11.4:8000}"

# 确保本地请求不走代理
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,7.150.11.4"
export NO_PROXY="${no_proxy}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# 从模型路径推导量化输出路径和结果路径
resolve_paths() {
    MODEL_PATH="$1"
    MODEL_NAME_BASE="$(basename "${MODEL_PATH}")"
    QUANT_OUTPUT="$(dirname "${MODEL_PATH}")/${MODEL_NAME_BASE}-RTN-W8A8"
    RESULTS_DIR="$(dirname "${TASK_DIR}")/results"
    RESULTS_FP16="${RESULTS_DIR}/${MODEL_NAME_BASE}-rtn-fp16"
    RESULTS_W8A8="${RESULTS_DIR}/${MODEL_NAME_BASE}-rtn-w8a8"
}

# 自动检测模型类型 → omni-npu patches 目录名
detect_model_type() {
    python3 -c "
import json, os
cfg = os.path.join('$1', 'config.json')
with open(cfg) as f:
    print(json.load(f).get('model_type', 'auto').lower())
" 2>/dev/null || echo "auto"
}

# 修正 yaml 中的数据路径 (只执行一次)
fix_yaml_path() {
    if grep -q '__TASK_DIR__' "${TASK_DIR}/wikitext_local.yaml"; then
        sed -i "s|__TASK_DIR__|${TASK_DIR}|g" "${TASK_DIR}/wikitext_local.yaml"
        info "Fixed dataset path → ${TASK_DIR}/data/wikitext2_doc_level"
    fi
}

# 等待 vllm serve 就绪
wait_for_serve() {
    info "Waiting for vllm serve to be ready at ${API_BASE} ..."
    for i in $(seq 1 180); do
        if curl --noproxy '*' -s "${API_BASE}/v1/models" > /dev/null 2>&1; then
            info "vllm serve is ready!"
            return 0
        fi
        sleep 2
    done
    error "vllm serve failed to start within 360 seconds. Check logs."
}

# ---- quantize ----
do_quantize() {
    [ -z "${1:-}" ] && error "Usage: rtn_eval.sh quantize /path/to/model"
    resolve_paths "$1"
    info "=== RTN W8A8 Quantize (CPU, pure safetensors) ==="
    info "Input:  ${MODEL_PATH}"
    info "Output: ${QUANT_OUTPUT}"

    if [ -d "${QUANT_OUTPUT}" ] && [ -n "$(ls "${QUANT_OUTPUT}"/*.safetensors 2>/dev/null)" ]; then
        warn "Already exists: ${QUANT_OUTPUT}, skipping"
        return 0
    fi

    # 纯 CPU 量化，不需要 NPU / torch_npu
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python3 "${TASK_DIR}/quantize_safetensors_int8.py" \
        --model "${MODEL_PATH}" --output "${QUANT_OUTPUT}"
    info "Quantize done → ${QUANT_OUTPUT}"
}

# ---- serve (omni-infer) ----
do_serve() {
    [ -z "${1:-}" ] && error "Usage: rtn_eval.sh serve /path/to/model"
    local model_path="$1"

    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

    # 检测服务是否已在运行
    if curl --noproxy '*' -s "${API_BASE}/v1/models" > /dev/null 2>&1; then
        info "Service already running at ${API_BASE}, skipping launch"
        return 0
    fi

    local model_type
    model_type=$(detect_model_type "${model_path}")

    export OMNI_NPU_PATCHES_DIR="${OMNI_NPU_PATCHES_DIR:-${model_type}}"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"

    info "=== Starting vllm serve (omni-infer) ==="
    info "Model: ${model_path}"
    info "Model type: ${model_type}"
    info "Port: ${SERVE_PORT}"
    info "TP size: ${TP_SIZE}"

    # 盘古模型: 使用 run_pangu.sh
    if [ "${model_type}" = "pangu_v2_moe" ]; then
        local pangu_script="${RUN_PANGU_SCRIPT:-}"
        # 自动查找 run_pangu.sh
        if [ -z "${pangu_script}" ]; then
            for candidate in \
                /home/p00929643/omni-npu/start_server/run_pangu.sh \
                "${TASK_DIR}/run_pangu.sh" \
                "$(dirname "${TASK_DIR}")/omni-npu/start_server/run_pangu.sh"; do
                if [ -f "${candidate}" ]; then
                    pangu_script="${candidate}"
                    break
                fi
            done
        fi
        [ -z "${pangu_script}" ] && error "run_pangu.sh not found. Set RUN_PANGU_SCRIPT=/path/to/run_pangu.sh"

        info "Using run_pangu.sh: ${pangu_script}"
        bash "${pangu_script}"
        wait_for_serve
    else
        # 通用模型: 直接启动 vllm serve
        python3 -m vllm.entrypoints.openai.api_server \
            --model "${model_path}" \
            --dtype auto \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --tensor-parallel-size "${TP_SIZE}" \
            --trust-remote-code \
            --host 0.0.0.0 \
            --port "${SERVE_PORT}" \
            ${EXTRA_SERVE_ARGS:-} \
            > "${TASK_DIR}/vllm_serve.log" 2>&1 &

        SERVE_PID=$!
        echo "${SERVE_PID}" > "${TASK_DIR}/.serve_pid"
        info "Server started (PID: ${SERVE_PID})"
        wait_for_serve
    fi
}

# ---- stop ----
do_stop() {
    info "=== Stopping vllm serve ==="
    if [ -f "${TASK_DIR}/.serve_pid" ]; then
        local pid
        pid=$(cat "${TASK_DIR}/.serve_pid")
        kill "${pid}" 2>/dev/null || true
        sleep 2
        kill -9 "${pid}" 2>/dev/null || true
        rm -f "${TASK_DIR}/.serve_pid"
    fi
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "APIServer" 2>/dev/null || true
    sleep 1
    info "vllm serve stopped"
}

# ---- eval (通过 API) ----
do_eval() {
    [ -z "${1:-}" ] && error "Usage: rtn_eval.sh eval /path/to/model [output_dir]"
    local model_path="$1"
    local output_dir="${2:-$(dirname "${TASK_DIR}")/results/$(basename "${model_path}")}"

    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    fix_yaml_path

    # 从服务获取实际注册的模型名
    local served_model
    served_model=$(curl --noproxy '*' -s "${API_BASE}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "${model_path}")

    info "=== PPL Eval via API ==="
    info "Model: ${model_path}"
    info "Served model name: ${served_model}"
    info "API: ${API_BASE}/v1/completions"
    info "Output: ${output_dir}"

    mkdir -p "${output_dir}"
    HF_DATASETS_OFFLINE=1 \
    no_proxy="*" NO_PROXY="*" \
    lm_eval --model local-completions \
        --model_args "model=${served_model},base_url=${API_BASE}/v1/completions,tokenizer_backend=huggingface,tokenizer=${model_path},trust_remote_code=True" \
        --include_path "${TASK_DIR}" \
        --tasks wikitext_local \
        --batch_size auto \
        --output_path "${output_dir}"

    info "Eval done → ${output_dir}"
}

# ---- eval_fp16: serve → eval (不自动 stop，服务可能是外部拉起的) ----
do_eval_fp16() {
    [ -z "${1:-}" ] && error "Usage: rtn_eval.sh eval_fp16 /path/to/model"
    resolve_paths "$1"
    info "====== FP16 Pipeline: serve → eval ======"
    do_serve "${MODEL_PATH}"
    do_eval "${MODEL_PATH}" "${RESULTS_FP16}" || true
}

# ---- eval_w8a8: serve → eval (不自动 stop) ----
do_eval_w8a8() {
    [ -z "${1:-}" ] && error "Usage: rtn_eval.sh eval_w8a8 /path/to/model"
    resolve_paths "$1"
    [ -d "${QUANT_OUTPUT}" ] || error "Quantized model not found: ${QUANT_OUTPUT}. Run 'rtn_eval.sh quantize' first."
    info "====== W8A8 Pipeline: serve → eval ======"
    do_serve "${QUANT_OUTPUT}"
    do_eval "${QUANT_OUTPUT}" "${RESULTS_W8A8}" || true
}

# ---- main ----
case "${1:-help}" in
    quantize)  do_quantize "${2:-}" ;;
    serve)     do_serve "${2:-}" ;;
    eval)      do_eval "${2:-}" "${3:-}" ;;
    stop)      do_stop ;;
    eval_fp16) do_eval_fp16 "${2:-}" ;;
    eval_w8a8) do_eval_w8a8 "${2:-}" ;;
    all)
        [ -z "${2:-}" ] && error "Usage: rtn_eval.sh all /path/to/model"
        do_quantize "$2"
        do_stop  # 确保没有残留进程
        do_eval_fp16 "$2"
        do_stop
        do_eval_w8a8 "$2"
        do_stop
        resolve_paths "$2"
        info "========================================="
        info "All done! Results:"
        info "  FP16: ${RESULTS_FP16}"
        info "  W8A8: ${RESULTS_W8A8}"
        info "========================================="
        ;;
    *)
        echo "Usage: bash rtn_eval.sh <command> [model_path]"
        echo ""
        echo "Commands:"
        echo "  quantize  /path/to/model   RTN W8A8 quantization (CPU, no GPU needed)"
        echo "  serve     /path/to/model   Start vllm serve (omni-infer)"
        echo "  eval      /path/to/model   Eval PPL via API (serve must be running)"
        echo "  stop                       Stop vllm serve"
        echo "  eval_fp16 /path/to/model   FP16: serve + eval"
        echo "  eval_w8a8 /path/to/model   W8A8: serve + eval"
        echo "  all       /path/to/model   Quantize + FP16 + W8A8"
        echo ""
        echo "Environment variables:"
        echo "  API_BASE              API address (default: http://7.150.11.4:8000)"
        echo "  SERVE_PORT            vllm serve port (default: 8000)"
        echo "  TP_SIZE               tensor parallel size (default: 1)"
        echo "  RUN_PANGU_SCRIPT      path to run_pangu.sh for pangu models"
        echo "  ASCEND_RT_VISIBLE_DEVICES  NPU devices (default: 0)"
        echo "  EXTRA_SERVE_ARGS      extra args for vllm serve"
        echo ""
        echo "Examples:"
        echo "  # Qwen3-0.6B (single NPU)"
        echo "  API_BASE=http://localhost:8000 bash rtn_eval.sh all /models/Qwen3-0.6B"
        echo ""
        echo "  # PanGu 92B (uses run_pangu.sh automatically)"
        echo "  bash rtn_eval.sh all /data/weights/pangu_v2/92B/iter_0059000_hf/"
        ;;
esac
