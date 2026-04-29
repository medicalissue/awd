#!/usr/bin/env bash
# Bare-EC2 / dstack worker bootstrap. Invoked by the VM's user-data at
# first boot.
#
# Pre-conditions:
#   * Ubuntu 22.04 + NVIDIA driver pre-installed (AWS Deep Learning Base
#     AMI works; vanilla Ubuntu needs a separate driver install step).
#   * IAM instance profile attached (S3 R/W on $CKPT_BUCKET, optional
#     ec2:TerminateInstances for self-cleanup).
#
# Responsibilities:
#   1. (Optional) mount EBS snapshot at /data — only when DATA_SNAPSHOT
#      is wired through. CIFAR is small enough to download on every
#      worker; the snapshot path exists for parity with the DyF/NELU
#      flow and for users that mirror datasets in EBS.
#   2. Clone the repo at /workspace and check out $REPO_REF.
#   3. Install Python deps (timm-style stack: torch, torchvision, wandb,
#      pyyaml, awscli, tensorboard).
#   4. Hand off to scripts/orchestrate.sh.
#   5. On orchestrator exit, self-terminate via ec2:TerminateInstances.
#
# Logs go to /var/log/awd/bootstrap.log AND /dev/console so
# `aws ec2 get-console-output` is useful when SSH is unreachable.
#
# Required env (exported by user-data wrapper):
#   REPO_URL            https://github.com/<user>/awd.git
#   REPO_REF            branch / tag / commit
#   CKPT_BUCKET         s3://awd-checkpoints
#   AWS_DEFAULT_REGION  us-west-2
# Optional:
#   WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
#   JOB_ORDER           override default queue
#   ENTRY_SCRIPT        defaults to scripts/orchestrate.sh
#   VENV_S3_URL         pre-built venv tarball (skips heavy pip install)

LOGDIR=/var/log/awd
mkdir -p "$LOGDIR"
LOG="$LOGDIR/bootstrap.log"
exec > >(tee -a "$LOG" | tee /dev/console) 2>&1

echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting on $(hostname)"

: "${REPO_URL:?REPO_URL required}"
: "${REPO_REF:=main}"
: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${WORKSPACE:=/workspace}"
# Optional pre-built venv tarball. If unset, we pip-install the small
# CIFAR stack into the system Python (~2 minutes on a DLBase AMI).
: "${VENV_S3_URL:=}"
: "${VENV_ROOT:=/opt/awd-venv}"
: "${DATA_MOUNT:=/data/cifar}"
export AWS_DEFAULT_REGION

die() {
    echo "[bootstrap] FATAL: $1 (rc=${2:-99})"
    exit "${2:-99}"
}

step() {
    local name="$1"; shift
    local logfile="${LOGDIR}/${name}.log"
    echo "[bootstrap] ▶ $name"
    "$@" > "$logfile" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        echo "[bootstrap] ✗ $name failed (rc=$rc). Last 40 lines of $logfile:"
        tail -40 "$logfile"
        return $rc
    fi
    echo "[bootstrap] ✓ $name ok"
}

# ── 1. Data mount (optional) ──────────────────────────────────────
# CIFAR is downloaded by torchvision on first run, so an EBS snapshot
# is *not* required. We mkdir DATA_MOUNT and move on. If the user
# wires up DATA_SNAPSHOT (via run_worker.sh's BlockDeviceMappings),
# the volume gets attached at /dev/sdg and we mount it here.
mount_data() {
    if [[ -z "${DATA_SNAPSHOT:-}" ]]; then
        mkdir -p "$DATA_MOUNT"
        echo "[bootstrap] no DATA_SNAPSHOT set — using local $DATA_MOUNT (torchvision will download)"
        return 0
    fi
    if mountpoint -q "$DATA_MOUNT"; then
        echo "[bootstrap] $DATA_MOUNT already mounted"; return 0
    fi
    local dev="" root_disk
    root_disk=$(findmnt -no SOURCE / | sed -E 's|p?[0-9]+$||')
    for _ in {1..30}; do
        while read -r name serial; do
            local devpath="/dev/$name"
            [[ "$devpath" == "$root_disk" ]] && continue
            [[ "$serial" != vol* ]] && continue
            if lsblk -no MOUNTPOINT "$devpath" | grep -qE '\S'; then
                continue
            fi
            dev="$devpath"
            break
        done < <(lsblk -dno NAME,SERIAL,TYPE | awk '$3=="disk" {print $1, $2}')
        [[ -n "$dev" ]] && break
        sleep 1
    done
    [[ -z "$dev" ]] && die "could not resolve $DATA_MOUNT device" 2
    echo "[bootstrap] mounting $dev at $DATA_MOUNT (rw)"
    mkdir -p "$DATA_MOUNT"
    mount "$dev" "$DATA_MOUNT" || die "mount $dev $DATA_MOUNT failed" 3
    df -h "$DATA_MOUNT"
}
mount_data || die "data mount failed" 9

# ── 2. Clone repo ─────────────────────────────────────────────────
if [[ -d "$WORKSPACE/.git" ]]; then
    step repo-update bash -c "cd $WORKSPACE && git fetch --all -q && \
        git checkout $REPO_REF -q && git pull -q" || die "repo-update failed" 6
else
    step repo-clone git clone -q --branch "$REPO_REF" "$REPO_URL" "$WORKSPACE" \
        || die "repo-clone failed" 6
fi

# ── 3. Install Python env ────────────────────────────────────────
if [[ -n "$VENV_S3_URL" ]]; then
    if [[ -x "${VENV_ROOT}/bin/python3.10" || -x "${VENV_ROOT}/bin/python3" ]]; then
        echo "[bootstrap] reusing existing ${VENV_ROOT}"
    else
        step fetch-venv aws s3 cp "$VENV_S3_URL" /tmp/awd-venv.tar.gz \
            || die "fetch-venv failed" 4
        step extract-venv bash -c \
            "mkdir -p /opt && tar xzf /tmp/awd-venv.tar.gz -C /opt && rm -f /tmp/awd-venv.tar.gz" \
            || die "extract-venv failed" 5
    fi
    if [[ ! -e "${VENV_ROOT}/bin/python" ]]; then
        if [[ -x "${VENV_ROOT}/bin/python3.10" ]]; then
            ln -sf python3.10 "${VENV_ROOT}/bin/python"
        elif [[ -x "${VENV_ROOT}/bin/python3" ]]; then
            ln -sf python3 "${VENV_ROOT}/bin/python"
        fi
    fi
    # shellcheck disable=SC1090
    source "${VENV_ROOT}/bin/activate"
else
    # No pre-built venv — pip-install into whatever python3 is on PATH.
    # The DL Base AMI ships with a usable PyTorch already.
    echo "[bootstrap] no VENV_S3_URL — using system python3"
fi

step verify-env python3 -c \
    "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || die "torch import failed" 7

step pip-deps python3 -m pip install --quiet --upgrade \
    torchvision pyyaml awscli wandb tensorboard \
    || die "pip install failed" 8

# ── 4. Orchestrate ────────────────────────────────────────────────
: "${ENTRY_SCRIPT:=scripts/orchestrate.sh}"
echo "[bootstrap] handing off to ${ENTRY_SCRIPT}"
cd "$WORKSPACE"
bash "$ENTRY_SCRIPT"
ORC_RC=$?
echo "[bootstrap] ${ENTRY_SCRIPT} exited rc=$ORC_RC"

# ── 5. Self-terminate ────────────────────────────────────────────
echo "[bootstrap] self-terminating spot instance"
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
IID=$(curl -sSH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
terminated=0
if [[ -n "$IID" ]]; then
    if aws ec2 terminate-instances --instance-ids "$IID" \
            --region "$AWS_DEFAULT_REGION" >>"$LOG" 2>&1; then
        echo "[bootstrap] terminate-instances OK for $IID"
        terminated=1
    else
        echo "[bootstrap] terminate-instances FAILED — scheduling OS halt"
    fi
fi
if (( terminated == 0 )); then
    shutdown -h +2 "awd worker self-terminate" >/dev/null 2>&1 || \
        (sleep 120 && halt -p) &
fi
exit $ORC_RC
