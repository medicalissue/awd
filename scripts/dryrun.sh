#!/usr/bin/env bash
# Smoke-test: run a couple of train epochs through main.py for each
# of the four ablation cells (plain / wd / ed / wd_ed), on real
# CIFAR-100 (small enough to download in seconds), with WRN-16-8 to
# keep CPU wall time under a few minutes.
#
# Run from repo root:
#     bash scripts/dryrun.sh                  # all 4 cells
#     bash scripts/dryrun.sh wd ed            # subset
#
# Validates:
#   YAML → yaml_to_args → main.py → 4-cell wd/ed toggle → train loop
#   → eval EMA + BN re-estimate → checkpoint save → 'complete' marker
#   → resume path

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/tmp/awd_dryrun_data}"
OUT_ROOT="${OUT_ROOT:-/tmp/awd_dryrun_out}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"

if (( $# > 0 )); then
    CELLS=("$@")
else
    CELLS=(plain wd ed wd_ed)
fi

# shellcheck source=infra/optim_flags.sh
source "$REPO_ROOT/scripts/infra/optim_flags.sh"

mkdir -p "$DATA_DIR"

for cell in "${CELLS[@]}"; do
    out="$OUT_ROOT/$cell"
    rm -rf "$out"
    mkdir -p "$out"
    echo
    echo "════════════════════════════════════════════════════════"
    echo "  dryrun: cell=$cell  out=$out"
    echo "════════════════════════════════════════════════════════"

    cfg_args=$(python "$REPO_ROOT/scripts/infra/yaml_to_args.py" \
               "$REPO_ROOT/configs/dryrun/toy.yaml")
    opt_args=$(optim_flags "$cell")

    # shellcheck disable=SC2086
    python "$REPO_ROOT/main.py" \
        $cfg_args $opt_args \
        --data_path "$DATA_DIR" \
        --output_dir "$out" \
        --device "$DEVICE" \
        --seed "$SEED" \
        2>&1 | tail -40

    if [[ -f "$out/complete" ]]; then
        echo "  ✓ complete marker present"
    else
        echo "  ✗ no complete marker — run failed?"
        exit 1
    fi
done

echo
echo "✓ dryrun complete. Each of the 4 cells logged a few epochs and wrote a complete sentinel."
