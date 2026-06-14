#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

eocrc-dat train --experiment "${EXPERIMENT}" "$@"
