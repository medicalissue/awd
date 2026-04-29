#!/usr/bin/env bash
# Map an optimizer-cell key → main.py CLI flags.
#
# Used by orchestrate_slot.sh and dryrun.sh. Single source of truth so
# that adding a new cell (e.g. ed at a different λ) is one case-arm
# here, not an N-place edit across every launcher.
#
# The headline 4-cell ablation (wd × ed) for CIFAR-100 / WRN-28x10:
#
#   plain   wd=0   ed=0       baseline minus all regularization
#   wd      wd=5e-4 ed=0      cs-giung / SWA paper baseline
#   ed      wd=0   ed=0.1     anchored decay alone
#   wd_ed   wd=5e-4 ed=0.1    both decays composed (paper main)
#
# Additional rows for ablation phases (λ_ed sweep, ED-only with various
# anchors, AdamE on the ViT lane) extend below.

optim_flags() {
    local key="$1"
    case "$key" in
        # ── Main 4 cells (CIFAR / SGD) ──────────────────────────────
        plain)
            printf '%s' "--optimizer sgd --weight_decay 0.0 --ed_lambda 0.0"
            ;;
        wd)
            printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 0.0"
            ;;
        ed)
            # λ_ed = 5e-4 (≡ wd magnitude) — early-training EMA anchor is
            # near θ_0, so the ed pull behaves like wd at that scale.
            # Picking λ_ed = λ_wd makes the four cells "same total
            # decay strength, different anchor source".
            printf '%s' "--optimizer sgd --weight_decay 0.0 --ed_lambda 5e-4 --anchor ema"
            ;;
        wd_ed)
            printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 5e-4 --anchor ema"
            ;;

        # ── λ_ed sensitivity sweep (anchor=ema, wd=5e-4) ────────────
        wd_ed-l1e4)  printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 1e-4 --anchor ema" ;;
        wd_ed-l3e4)  printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 3e-4 --anchor ema" ;;
        wd_ed-l1e3)  printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 1e-3 --anchor ema" ;;
        wd_ed-l3e3)  printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 3e-3 --anchor ema" ;;

        # ── Anchor alternatives at the chosen λ_ed = 5e-4 ───────────
        wd_ed-init)    printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 5e-4 --anchor init" ;;
        wd_ed-polyak)  printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 5e-4 --anchor polyak" ;;
        wd_ed-window)  printf '%s' "--optimizer sgd --weight_decay 5e-4 --ed_lambda 5e-4 --anchor window --window 16" ;;

        # ── AdamE (ViT lane) — kept for the secondary table ─────────
        adamw)
            printf '%s' "--optimizer adamw --weight_decay 0.05"
            ;;
        adame-ema)
            printf '%s' "--optimizer adame --weight_decay 0.05 --anchor ema"
            ;;
        *)
            echo "ERROR: unknown optimizer cell '$key'." \
                "See scripts/infra/optim_flags.sh." >&2
            return 2
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    optim_flags "$@"
fi
