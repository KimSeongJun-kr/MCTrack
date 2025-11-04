#!/usr/bin/env bash
set -euo pipefail

# Keep terminal window open when script finishes (success or error)
pause_if_terminal() {
  if [[ -t 1 ]]; then
    exec "${SHELL:-/bin/bash}" -l
  fi
}

on_error() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    echo "스크립트가 오류 코드 $status 로 종료되었습니다."
  fi
}

trap 'on_error' ERR
trap 'pause_if_terminal' EXIT

# Defaults
BASE_DIR="/3dmot_ws/MCTrack"
PROCESSORS=""
SPLIT=""
STEP=""
TIMESTAMP=""
DETS_PATH=""

usage() {
  echo "Usage: $0 \
  --timestamp YYYYMMDD_hhmmss \
  --step N \
  --processors N \
  --split {val|test|train} \
  --dets_path /abs/path.json"
  echo ""
  echo "Examples:"
  echo "  $0 \
  --timestamp 20251104_170700 \
  --step 1 \
  --processors 20 \
  --split val \
  --dets_path /3dmot_ws/MCTrack/data/nuscenes/detectors/centerpoint/val.json"
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split|-s) SPLIT="$2"; shift 2;;
    --step|-st) STEP="$2"; shift 2;;
    --processors|-p) PROCESSORS="$2"; shift 2;;
    --timestamp|-t) TIMESTAMP="$2"; shift 2;;
    --dets_path|-dp) DETS_PATH="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

MISSING_ARGS=()
[[ -z "$TIMESTAMP" ]] && MISSING_ARGS+=("--timestamp")
[[ -z "$STEP" ]] && MISSING_ARGS+=("--step")
[[ -z "$PROCESSORS" ]] && MISSING_ARGS+=("--processors")
[[ -z "$SPLIT" ]] && MISSING_ARGS+=("--split")
[[ -z "$DETS_PATH" ]] && MISSING_ARGS+=("--dets_path")

if [[ ${#MISSING_ARGS[@]} -gt 0 ]]; then
  echo "Error: missing required arguments: ${MISSING_ARGS[*]}"
  usage
  exit 1
fi

# Validate values
if [[ ! "$TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "Error: --timestamp must match YYYYMMDD_hhmmss (e.g., 20251104_170700)"
  exit 1
fi

if [[ ! "$STEP" =~ ^[0-9]+$ ]]; then
  echo "Error: --step must be an integer"
  exit 1
fi

if [[ ! "$PROCESSORS" =~ ^[0-9]+$ ]]; then
  echo "Error: --processors must be an integer"
  exit 1
fi

case "$SPLIT" in
  val|test|train) ;;
  *) echo "Error: --split must be one of {val|test|train}"; exit 1;;
esac

if [[ ! -f "$DETS_PATH" ]]; then
  echo "Error: --dets_path not found: $DETS_PATH"
  exit 1
fi

SAVE_BASE_VERSION="${BASE_DIR}/data/base_version/nuscenes/${TIMESTAMP}/step_${STEP}"
RESULT_SAVE="${BASE_DIR}/prsys_results/${TIMESTAMP}/step_${STEP}"

echo "[1/2] Converting detections to base version"
echo "python ${BASE_DIR}/preprocess/convert2baseversion.py --dets_path $DETS_PATH --save_path $SAVE_BASE_VERSION --split $SPLIT"
python "${BASE_DIR}/preprocess/convert2baseversion.py" \
  --dets_path "$DETS_PATH" \
  --save_path "$SAVE_BASE_VERSION" \
  --split "$SPLIT"

echo "[2/2] Running main tracking pipeline"
echo "python ${BASE_DIR}/main.py -p $PROCESSORS --dets_path $SAVE_BASE_VERSION --save_path $RESULT_SAVE --split $SPLIT"
python "${BASE_DIR}/main.py" -p "$PROCESSORS" \
  --dets_path "$SAVE_BASE_VERSION" \
  --save_path "$RESULT_SAVE" \
  --split "$SPLIT"

echo "Done. Results saved to: $RESULT_SAVE"


