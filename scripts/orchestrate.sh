#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# AWD CIFAR worker fanout.
#
# Each CIFAR job uses a single GPU. A multi-GPU VM (g5.12xlarge =
# 4×A10G) should pull ``ngpus`` independent jobs at once from the
# shared S3 queue. This script spawns one ``orchestrate_slot.sh`` per
# visible GPU, each pinned to its own CUDA_VISIBLE_DEVICES, and waits
# for all of them. Slot runners race against the same lease queue:
# whichever GPU finishes first pops the next job. Order is naturally
# "first-idle-first-serve".
#
# Shared-queue coordination lives entirely in the slot script; see
# scripts/orchestrate_slot.sh for the lease-claim / heartbeat /
# preempt logic. Environment variables propagate to every slot.
#
# CPU fallback: if no GPU is visible, fall back to a single slot.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { printf '[orchestrate %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLOT_RUNNER="$SCRIPT_DIR/orchestrate_slot.sh"

# Detect GPUs. Honor an explicit override for debugging:
#     NUM_AWD_SLOTS=2 bash orchestrate.sh
if [[ -n "${NUM_AWD_SLOTS:-}" ]]; then
    ngpus="$NUM_AWD_SLOTS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    ngpus=$(nvidia-smi -L | wc -l | tr -d ' ')
else
    ngpus=1
fi
if (( ngpus < 1 )); then ngpus=1; fi

log "spawning $ngpus slot runner(s)"

pids=()
for (( gpu=0; gpu<ngpus; gpu++ )); do
    # Stagger slot starts by `gpu` seconds. Four slots firing their
    # S3 lease claim at the same wall-clock millisecond is the root
    # cause of the concurrent-claim race that the read-back
    # confirmation step also guards against — but a few seconds of
    # jitter avoids the read-back work in the common case.
    (
        sleep "$gpu"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export AWD_GPU_SLOT="$gpu"
        exec bash "$SLOT_RUNNER"
    ) &
    pids+=("$!")
    log "  slot $gpu (CUDA_VISIBLE_DEVICES=$gpu) pid=${pids[-1]}"
done

# Forward SIGTERM/SIGINT to every child. AWS spot preempt → bootstrap
# wrapper SIGTERMs us, we forward to each slot which forwards to its
# trainer via the per-slot preempt watcher.
cleanup() {
    log "received signal — forwarding to all slots"
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup TERM INT

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
done

log "all slots exited (rc=$rc)"
exit "$rc"
