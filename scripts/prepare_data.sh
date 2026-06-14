#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-data/synthetic}"
SEED="${2:-7}"

eocrc-dat infer --describe_cohort --data synthetic --seed "${SEED}" --out "${OUT}"
