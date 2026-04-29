#!/usr/bin/env bash
# Spot-fleet watchdog. Keeps TARGET_WORKERS live in the AZs listed in
# CAMPAIGN_AZS until every experiment has a `complete` sentinel in S3.
#
# Behavior:
#   * Maintain TARGET_WORKERS spot instances tagged Project=awd,
#     Role=worker, Campaign=$CAMPAIGN. If fewer are live, launch one
#     by rotating through CAMPAIGN_AZS in user-specified order.
#   * Completion: when every experiment key in JOB_ORDER has a
#     `complete` object under $CKPT_BUCKET/<exp>/, exit 0.
#   * Capacity retry: if run-instances fails in all CAMPAIGN_AZS, sleep
#     POLL_INTERVAL_SEC and try again — forever. Never escalates to
#     on-demand, never gives up.
#
# Required env (.env):
#   CKPT_BUCKET, WANDB_API_KEY, JOB_ORDER, CAMPAIGN_AZS,
#   AMI, SG, IAM_PROFILE, SUBNET_<AZ>
# Optional:
#   TARGET_WORKERS=2
#   INSTANCE_TYPE=g5.xlarge
#   POLL_INTERVAL_SEC=60
#   CAPACITY_SLEEP_SEC=60
#   CAMPAIGN=awd
set -euo pipefail

: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required (empty string to disable W&B)}"
: "${JOB_ORDER:?JOB_ORDER required}"
: "${CAMPAIGN_AZS:?CAMPAIGN_AZS required}"
: "${TARGET_WORKERS:=1}"
: "${INSTANCE_TYPE:=g5.12xlarge}"
: "${POLL_INTERVAL_SEC:=60}"
: "${CAPACITY_SLEEP_SEC:=60}"
: "${CAMPAIGN:=awd}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
export CKPT_BUCKET WANDB_API_KEY CAMPAIGN AWS_DEFAULT_REGION

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WORKER="$SCRIPT_DIR/run_worker.sh"

log() { printf '[watchdog %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

bucket_root="${CKPT_BUCKET%/}"
bucket_name=$(echo "$bucket_root" | sed -E 's|^s3://([^/]+).*|\1|')
bucket_prefix=$(echo "$bucket_root" | sed -E 's|^s3://[^/]+/?||')
[[ -n "$bucket_prefix" ]] && bucket_prefix="${bucket_prefix}/"

exp_complete() {
    local exp="$1"
    local key_flat="${bucket_prefix}${exp}/complete"
    aws s3api head-object --bucket "$bucket_name" --key "$key_flat" \
        >/dev/null 2>&1
}

_exp_from_entry() {
    # JOB_ORDER entries are <cfg>:<cell>:<seed> triples (CIFAR slot
    # convention). Older callers may still pass <cfg>:<cell> pairs;
    # both shapes parse the same way — empty seed just yields a
    # trailing dash that the slot's exp_key produces too.
    local entry="$1" cfg cell seed
    IFS=: read -r cfg cell seed <<<"$entry"
    local base
    base=$(basename "${cfg%.yaml}")
    if [[ -n "$seed" ]]; then
        echo "${base}-${cell}-s${seed}"
    else
        echo "${base}-${cell}"
    fi
}

all_done() {
    for entry in $JOB_ORDER; do
        local exp; exp=$(_exp_from_entry "$entry")
        exp_complete "$exp" || return 1
    done
    return 0
}

count_incomplete() {
    local n=0
    for entry in $JOB_ORDER; do
        local exp; exp=$(_exp_from_entry "$entry")
        exp_complete "$exp" || n=$((n + 1))
    done
    echo "$n"
}

count_live_workers() {
    aws ec2 describe-instances \
        --region "$AWS_DEFAULT_REGION" \
        --filters \
            "Name=tag:Project,Values=awd" \
            "Name=tag:Role,Values=worker" \
            "Name=tag:Campaign,Values=${CAMPAIGN}" \
            "Name=instance-state-name,Values=pending,running" \
        --query 'length(Reservations[].Instances[])' \
        --output text
}

launch_one() {
    local az_cycle=($CAMPAIGN_AZS)
    local cycle=0
    while :; do
        for az in "${az_cycle[@]}"; do
            local suffix
            suffix="$(date -u +%Y%m%dT%H%M%S)-${az##*-}"
            log "launching worker in $az ($INSTANCE_TYPE)"
            if bash "$RUN_WORKER" "$az" "$INSTANCE_TYPE" "$suffix" \
                    > /tmp/awd-launch-$$.log 2>&1; then
                log "launch OK"
                head -10 /tmp/awd-launch-$$.log
                rm -f /tmp/awd-launch-$$.log
                return 0
            fi
            local err; err=$(tail -5 /tmp/awd-launch-$$.log | tr '\n' ' ')
            log "launch in $az failed: $err"
        done
        cycle=$((cycle + 1))
        log "AZ cycle $cycle exhausted — sleeping ${CAPACITY_SLEEP_SEC}s and retrying"
        sleep "$CAPACITY_SLEEP_SEC"
    done
}

log "starting. campaign=$CAMPAIGN target=$TARGET_WORKERS type=$INSTANCE_TYPE AZs=($CAMPAIGN_AZS)"
while :; do
    if all_done; then
        log "queue drained — every experiment has a complete sentinel. exiting."
        break
    fi

    live=$(count_live_workers)
    remaining=$(count_incomplete)
    effective_target=$TARGET_WORKERS
    (( remaining < effective_target )) && effective_target=$remaining
    log "live: $live / $effective_target  (TARGET=$TARGET_WORKERS, remaining_jobs=$remaining)"

    while (( live < effective_target )); do
        if launch_one; then
            live=$((live + 1))
        fi
    done

    sleep "$POLL_INTERVAL_SEC"
done
