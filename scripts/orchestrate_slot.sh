#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# AWD CIFAR-100 slot runner (one process per GPU).
#
# Pulls jobs from the shared S3 lease queue and runs them one at a time
# on the GPU selected by ``CUDA_VISIBLE_DEVICES`` (set by the fanout
# script orchestrate.sh). The queue logic — lease claim, heartbeat,
# preempt forwarding, final sync — is borrowed from NELU's
# orchestrate_cifar_slot.sh, adapted to our (cfg, cell, seed) triple.
#
# Queue layout under $CKPT_BUCKET :
#
#     <CKPT_BUCKET>/<exp>/             # <exp> = <cfg>-<cell>-s<seed>
#         complete           sentinel; presence ⇒ done
#         lease              "<instance-id>-g<gpu> <unix-ts>"
#         checkpoint-N.pth   training rolling history
#         checkpoint.pt      stable symlink to the latest (NELU style)
#         log.txt, args.json
#
# The lease owner string includes the GPU slot suffix so a single VM
# running N slots can hold N distinct leases without races. Trainer
# pgid files are also per-slot.
#
# Required env (inherited from the fanout):
#   CUDA_VISIBLE_DEVICES, AWD_GPU_SLOT
#   CKPT_BUCKET, AWS_DEFAULT_REGION
# Optional:
#   WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
#   JOB_ORDER            whitespace-separated <cfg>:<cell>:<seed> triples
#   LEASE_TTL, HEARTBEAT_EVERY, MAX_IDLE_PASSES
#   DATA_MOUNT           default /data/cifar
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${AWD_GPU_SLOT:?AWD_GPU_SLOT must be set by the fanout script}"
SLOT="$AWD_GPU_SLOT"

log() {
    printf '[slot-%s %s] %s\n' "$SLOT" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

: "${CKPT_BUCKET:?CKPT_BUCKET not set (e.g. s3://awd-checkpoints)}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${WANDB_PROJECT:=awd}"
: "${WANDB_ENTITY:=}"
: "${LEASE_TTL:=600}"
: "${HEARTBEAT_EVERY:=60}"
: "${MAX_IDLE_PASSES:=1}"
: "${DATA_MOUNT:=/data/cifar}"
export AWS_DEFAULT_REGION

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${JOB_ORDER:-}" ]]; then
    job_file="$REPO_ROOT/scripts/infra/default_job_order.txt"
    if [[ -f "$job_file" ]]; then
        JOB_ORDER=$(grep -v '^\s*#' "$job_file" | grep -v '^\s*$' | tr '\n' ' ')
    fi
fi
: "${JOB_ORDER:?JOB_ORDER empty and default_job_order.txt missing}"

# Optimizer-cell → CLI flag mapping.
# shellcheck source=infra/optim_flags.sh
source "$REPO_ROOT/scripts/infra/optim_flags.sh"

# ── IMDSv2 helpers ─────────────────────────────────────────────────
imds() {
    local token
    token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
    curl -sS -H "X-aws-ec2-metadata-token: $token" \
         "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || true
}

AZ=$(imds placement/availability-zone)
INSTANCE_ID=$(imds instance-id)
[[ -z "$INSTANCE_ID" ]] && INSTANCE_ID="$(hostname)-$$"
# Lease owner must be unique per (VM, GPU) so multiple slots on the
# same VM don't clobber each other.
OWNER="${INSTANCE_ID}-g${SLOT}"
PGID_FILE="/tmp/trainer-g${SLOT}.pgid"
log "AZ=${AZ:-unknown} instance=$INSTANCE_ID owner=$OWNER"

# ── Data sanity ────────────────────────────────────────────────────
# CIFAR is small; torchvision will download into $DATA_MOUNT on first
# run. We just need the directory to exist and be writable.
check_data_mount() {
    if ! mkdir -p "$DATA_MOUNT" 2>/dev/null; then
        log "FATAL: cannot create $DATA_MOUNT"
        exit 4
    fi
    if [[ ! -w "$DATA_MOUNT" ]]; then
        log "FATAL: $DATA_MOUNT not writable"
        exit 4
    fi
    log "data mount OK: $DATA_MOUNT"
}

# ── Preempt watcher ────────────────────────────────────────────────
start_preempt_watcher() {
    (
        while :; do
            local token code
            token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
                    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
            code=$(curl -sS -o /dev/null -w '%{http_code}' \
                   -H "X-aws-ec2-metadata-token: $token" \
                   "http://169.254.169.254/latest/meta-data/spot/instance-action" \
                   2>/dev/null || echo "000")
            if [[ "$code" == "200" ]]; then
                if [[ -f "$PGID_FILE" ]]; then
                    pgid=$(cat "$PGID_FILE")
                    log "preempt notice — SIGTERM to pgid=$pgid"
                    kill -TERM -"$pgid" 2>/dev/null || true
                fi
                break
            fi
            sleep 5
        done
    ) &
    log "preempt watcher pid=$!"
}

# ── S3 helpers ─────────────────────────────────────────────────────
s3_exists() { aws s3 ls "$1" >/dev/null 2>&1; }

exp_key() {
    local cfg="$1" cell="$2" seed="$3"
    local base
    base=$(basename "${cfg%.yaml}")
    echo "${base}-${cell}-s${seed}"
}

lease_claim() {
    # Optimistic claim with a read-back confirmation step.
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner ts age
    now=$(date +%s)
    if s3_exists "$key"; then
        read -r owner ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
        ts=${ts:-0}
        age=$((now - ts))
        if (( age < LEASE_TTL )); then
            return 1
        fi
        log "stealing stale lease on $exp (age=${age}s, owner=$owner)"
    fi
    echo "$OWNER $now" | aws s3 cp - "$key" >/dev/null
    sleep "$(awk "BEGIN{print 0.5+rand()}")"
    local winner
    read -r winner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    if [[ "$winner" != "$OWNER" ]]; then
        log "lost lease race on $exp (winner=$winner, we=$OWNER)"
        return 1
    fi
    return 0
}

lease_refresh() {
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner
    now=$(date +%s)
    read -r owner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    [[ "$owner" == "$OWNER" ]] || return 1
    echo "$OWNER $now" | aws s3 cp - "$key" >/dev/null
}

lease_release() {
    aws s3 rm "${CKPT_BUCKET}/$1/lease" >/dev/null 2>&1 || true
}

# ── Run one job ────────────────────────────────────────────────────
run_job() {
    local cfg="$1" cell="$2" seed="$3"
    local exp s3_prefix
    exp=$(exp_key "$cfg" "$cell" "$seed")
    s3_prefix="${CKPT_BUCKET}/${exp}"

    if s3_exists "${s3_prefix}/complete"; then
        log "skip $exp (complete)"
        return 3
    fi

    if ! lease_claim "$exp"; then
        log "skip $exp (fresh lease held)"
        return 2
    fi

    log "▶ running $exp on GPU slot $SLOT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?})"

    local outdir="/tmp/runs/${exp}"
    mkdir -p "$outdir"

    # Pull prior state.
    aws s3 sync "${s3_prefix}/" "${outdir}/" \
        --exclude "lease" --exclude "complete" \
        --exact-timestamps --only-show-errors || true

    # Defense-in-depth: if S3 has any checkpoint but we didn't land
    # one, the sync silently failed. Bail loudly rather than restart
    # from scratch.
    local s3_has_ckpt=0
    if aws s3 ls "${s3_prefix}/checkpoint-" >/dev/null 2>&1; then
        s3_has_ckpt=1
    fi
    local local_has_ckpt=0
    if compgen -G "${outdir}/checkpoint-*.pth" >/dev/null 2>&1; then
        local_has_ckpt=1
    fi
    if (( s3_has_ckpt == 1 )) && (( local_has_ckpt == 0 )); then
        log "FATAL: S3 has checkpoints for ${exp} but local copy missing after sync"
        log "  s3_prefix=${s3_prefix}  outdir=${outdir}"
        ls -lR "$outdir" >&2 || true
        lease_release "$exp"
        rm -rf "$outdir"
        return 1
    fi

    # Build CLI args.
    local cfg_args opt_args
    cfg_args=$(python "$REPO_ROOT/scripts/infra/yaml_to_args.py" "$cfg")
    opt_args=$(optim_flags "$cell")

    local wandb_args=()
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        wandb_args+=(--enable_wandb true --wandb_project "$WANDB_PROJECT")
        if [[ -n "$WANDB_ENTITY" ]]; then
            wandb_args+=(--wandb_entity "$WANDB_ENTITY")
        fi
        if command -v wandb >/dev/null 2>&1; then
            wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
        fi
    fi

    # Heartbeat + interim S3 push.
    (
        while sleep "$HEARTBEAT_EVERY"; do
            if ! lease_refresh "$exp"; then
                log "  lease lost — stopping heartbeat"
                if [[ -f "$PGID_FILE" ]]; then
                    pgid=$(cat "$PGID_FILE")
                    kill -TERM -"$pgid" 2>/dev/null || true
                fi
                exit 0
            fi
            aws s3 sync "${outdir}/" "${s3_prefix}/" \
                --exclude "*.tmp" --exclude "lease" \
                --only-show-errors || true
        done
    ) &
    local heartbeat_pid=$!

    # Launch trainer in its own session so the preempt watcher can
    # SIGTERM the whole tree by pgid.
    local trainer_rc=0
    set +e
    # shellcheck disable=SC2086
    setsid python "$REPO_ROOT/main.py" \
        $cfg_args $opt_args \
        --seed "$seed" \
        --output_dir "$outdir" \
        --device cuda \
        "${wandb_args[@]}" \
        >>"$outdir/log.txt" 2>&1 &
    local trainer_pid=$!
    echo "$trainer_pid" >"$PGID_FILE"
    wait "$trainer_pid"
    trainer_rc=$?
    set -e

    kill "$heartbeat_pid" 2>/dev/null || true
    rm -f "$PGID_FILE"

    # Final sync.
    aws s3 sync "${outdir}/" "${s3_prefix}/" --exclude "lease" \
        --only-show-errors || true

    if [[ -f "$outdir/complete" ]]; then
        aws s3 cp "$outdir/complete" "${s3_prefix}/complete" \
            --only-show-errors || true
        log "✓ $exp complete"
    elif (( trainer_rc == 0 )); then
        date -u +%Y-%m-%dT%H:%M:%SZ \
            | aws s3 cp - "${s3_prefix}/complete" >/dev/null
        log "✓ $exp complete (rc=0)"
    elif (( trainer_rc == 143 || trainer_rc == 137 )); then
        log "⏸ $exp paused (signal exit rc=$trainer_rc, will resume)"
    else
        log "✗ $exp failed (rc=$trainer_rc) — see ${s3_prefix}/log.txt"
    fi

    lease_release "$exp"
    rm -rf "$outdir"
    return "$trainer_rc"
}

# ── Main ───────────────────────────────────────────────────────────
check_data_mount
start_preempt_watcher

: "${MAX_FAILS_PER_JOB:=2}"
declare -A fail_count

idle_passes=0
while (( idle_passes < MAX_IDLE_PASSES )); do
    ran_any=0
    for triple in $JOB_ORDER; do
        IFS=: read -r cfg cell seed <<<"$triple"
        if [[ -z "$cfg" || -z "$cell" || -z "$seed" ]]; then
            log "skipping malformed triple: $triple"
            continue
        fi
        exp=$(exp_key "$cfg" "$cell" "$seed")
        if (( ${fail_count[$exp]:-0} >= MAX_FAILS_PER_JOB )); then
            log "skip $exp (failed ${fail_count[$exp]}× already, blacklisted on this slot)"
            continue
        fi
        rc=0
        run_job "$cfg" "$cell" "$seed" || rc=$?
        case $rc in
            0)
                ran_any=1
                fail_count[$exp]=0
                ;;
            2|3) ;;  # lease held / complete → don't reset idle counter
            *)
                ran_any=1
                fail_count[$exp]=$(( ${fail_count[$exp]:-0} + 1 ))
                log "  consecutive failures on $exp: ${fail_count[$exp]}/${MAX_FAILS_PER_JOB}"
                ;;
        esac
    done
    if (( ran_any == 0 )); then
        idle_passes=$((idle_passes + 1))
        log "no work in this pass (${idle_passes}/${MAX_IDLE_PASSES})"
        sleep 10
    else
        idle_passes=0
    fi
done

log "queue drained — slot exiting"
