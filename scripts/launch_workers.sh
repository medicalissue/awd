#!/usr/bin/env bash
# dstack launch path. Submits N workers as separate dstack tasks.
#
# Each worker:
#   * runs .dstack/worker.dstack.yml (NGC PyTorch + repo + orchestrate.sh)
#   * pulls jobs from $CKPT_BUCKET via the S3 lease layer
#   * exits when the queue is drained or it gets preempted
#
# Workers are independent; the S3 lease keeps them from stepping on each
# other. Adding more after the first batch just means more parallel
# capacity.
#
# Usage:
#   source .env
#   bash scripts/launch_workers.sh 2          # 2 workers

set -euo pipefail

: "${CKPT_BUCKET:?source .env first (CKPT_BUCKET unset)}"
: "${WANDB_API_KEY:?WANDB_API_KEY unset (use empty string to disable W&B)}"
: "${WANDB_ENTITY:=}"

N="${1:-2}"

for i in $(seq 1 "$N"); do
    name="awd-worker-${i}"
    echo "▶ submitting ${name}"
    dstack apply -f .dstack/worker.dstack.yml \
        -P . \
        -n "${name}" \
        -e CKPT_BUCKET="${CKPT_BUCKET}" \
        -e WANDB_API_KEY="${WANDB_API_KEY}" \
        -e WANDB_ENTITY="${WANDB_ENTITY}" \
        -y -d
done

cat <<EOF

${N} dstack worker(s) submitted. Useful follow-ups:
    dstack ps                           # active runs
    dstack logs awd-worker-1 -f         # stream a worker's log
    aws s3 ls $CKPT_BUCKET/ --recursive | grep complete   # done list
EOF
