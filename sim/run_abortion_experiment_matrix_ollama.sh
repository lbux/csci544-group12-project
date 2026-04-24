#!/usr/bin/env bash
set -euo pipefail

# Run the 12-run abortion debate experiment matrix:
#   4 debate models x 3 simulation types
#
# Fixed moderator setup for the moderated Reddit simulation:
#   judge model        = llama3.1:8b
#   intervention model = llama3.1:8b
#   local classifier   = models/cga_deberta_onnx_int8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT="${INPUT:-out.jsonl}"
ROUNDS="${ROUNDS:-8}"
BASE_URL="${BASE_URL:-http://localhost:11434/v1/}"
API_KEY="${API_KEY:-ollama}"
JUDGE_MODEL="${JUDGE_MODEL:-llama3.1:8b}"
INTERVENTION_MODEL="${INTERVENTION_MODEL:-llama3.1:8b}"
CLASSIFIER="${CLASSIFIER:-models/cga_deberta_onnx_int8}"
TOXICITY_THRESHOLD="${TOXICITY_THRESHOLD:-0.6}"
INTERVENTION_THRESHOLD="${INTERVENTION_THRESHOLD:-10}"
COOLDOWN_TURNS="${COOLDOWN_TURNS:-2}"
THINKING="${THINKING:-true}"
LOG_DIR="${LOG_DIR:-sim_debate_records/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/abortion_experiment_matrix_$(date +%Y%m%d_%H%M%S).log}"
SIMULATION_TYPES=("naive" "reddit_aligned" "moderated_reddit_aligned")

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ "${THINKING}" == "true" ]]; then
  THINKING_ARG=(--thinking)
else
  THINKING_ARG=(--no-thinking)
fi

DEBATE_MODELS=(
  # "qwen3.5:9b"
  # "fredrezones55/Qwen3.5-Uncensored-HauhauCS-Aggressive:9b"
  # "gemma4:e4b"
  "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b"
)
TOTAL_RUNS=$((${#DEBATE_MODELS[@]} * ${#SIMULATION_TYPES[@]}))
RUN_INDEX=0

if [[ ! -f "${INPUT}" ]]; then
  echo "Input file not found: ${INPUT}" >&2
  exit 1
fi

echo "Experiment matrix (Ollama)"
echo "  input: ${INPUT}"
echo "  rounds: ${ROUNDS}"
echo "  base url: ${BASE_URL}"
echo "  api key: ${API_KEY}"
echo "  judge model: ${JUDGE_MODEL}"
echo "  intervention model: ${INTERVENTION_MODEL}"
echo "  classifier: ${CLASSIFIER}"
echo "  toxicity threshold: ${TOXICITY_THRESHOLD}"
echo "  intervention threshold: ${INTERVENTION_THRESHOLD}"
echo "  cooldown turns: ${COOLDOWN_TURNS}"
echo "  thinking: ${THINKING}"
echo "  log file: ${LOG_FILE}"
echo "  debate models (${#DEBATE_MODELS[@]}):"
for MODEL in "${DEBATE_MODELS[@]}"; do
  echo "    - ${MODEL}"
done
echo "  simulation types (${#SIMULATION_TYPES[@]}):"
for SIM_TYPE in "${SIMULATION_TYPES[@]}"; do
  echo "    - ${SIM_TYPE}"
done
echo "  total runs: ${TOTAL_RUNS}"
echo

for MODEL in "${DEBATE_MODELS[@]}"; do
  echo "============================================================"
  echo "Debate model: ${MODEL}"
  echo "============================================================"

  RUN_INDEX=$((RUN_INDEX + 1))
  echo "[run ${RUN_INDEX}/${TOTAL_RUNS}] [1/3] Naive debate"
  uv run python sim/naive_abortion_debate.py \
    --rounds "${ROUNDS}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    "${THINKING_ARG[@]}"

  RUN_INDEX=$((RUN_INDEX + 1))
  echo "[run ${RUN_INDEX}/${TOTAL_RUNS}] [2/3] Reddit aligned debate"
  uv run python sim/reddit_abortion_debate.py \
    --input "${INPUT}" \
    --rounds "${ROUNDS}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    "${THINKING_ARG[@]}"

  RUN_INDEX=$((RUN_INDEX + 1))
  echo "[run ${RUN_INDEX}/${TOTAL_RUNS}] [3/3] Moderated Reddit aligned debate"
  uv run python sim/moderation_reddit_abortion_debate.py \
    --input "${INPUT}" \
    --rounds "${ROUNDS}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    --judge-model "${JUDGE_MODEL}" \
    --intervention-model "${INTERVENTION_MODEL}" \
    --classifier "${CLASSIFIER}" \
    --toxicity-threshold "${TOXICITY_THRESHOLD}" \
    --intervention-threshold "${INTERVENTION_THRESHOLD}" \
    --cooldown-turns "${COOLDOWN_TURNS}" \
    "${THINKING_ARG[@]}"

  echo
done

echo "All experiment runs completed."
echo "Outputs are in sim_debate_records/"
echo "Full experiment log: ${LOG_FILE}"
