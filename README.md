# Anchored Weight Decay (AWD)

Codebase for the *Anchored Weight Decay* paper — a generalization of
weight decay where the regularizer pulls θ toward a chosen anchor point
instead of the origin:

    L̃(θ) = L(θ) + (λ/2) ‖θ − θ_anchor(t)‖²

Anchor variants implemented (all decoupled, à la AdamW):

| Anchor      | θ_anchor                                     | Notes                          |
| ----------- | -------------------------------------------- | ------------------------------ |
| `origin`    | 0                                            | ≡ AdamW (sanity-checked)       |
| `init`      | θ₀                                           | static, lazy-regime / L2-SP    |
| `ema`       | α·θ_anchor + (1−α)·θ                         | **AdamE — main method**        |
| `polyak`    | running mean over all steps                  | SWA-like coupling              |
| `window`    | mean of last *W* param snapshots             | recent-trajectory anchor       |

The paper's contribution is the framework, not just AdamE — see the
Related Work table in §4 for how this unifies AdamW, L2-SP, SWA,
Lookahead, and BYOL-style EMA targets.

---

## Layout

    main.py                        CIFAR-{10,100} training entrypoint
    awd/
        optim.py                   AdamE + anchor-aware param groups
        models.py                  CIFAR ResNet-18/34, WRN-28-10
        data.py                    CIFAR loaders (mean/std normalize)
        hessian.py                 Top-eigenvalue power iter (sharpness)
        utils.py                   dist init, ckpt I/O, cosine LR
    configs/
        _base.yaml                 shared defaults
        cifar100/                  primary experiment configs
        dryrun/toy.yaml            2-epoch smoke test
    scripts/
        orchestrate.sh             single-worker job-queue runner
        bootstrap.sh               EC2/dstack first-boot bootstrap
        dryrun.sh                  local smoke test
        launch_local.sh            spawn N workers locally
        launch_workers.sh          submit N dstack tasks
        launch_campaign.sh         bare-EC2 watchdog launcher
        infra/
            optim_flags.sh         variant-key → CLI-flag mapping
            yaml_to_args.py        YAML → argparse CLI flags
            default_job_order.txt  default <config>:<variant> queue
            watchdog.sh            spot-fleet keepalive
            run_worker.sh          single bare-EC2 spot launch
            setup_aws.sh           one-shot S3+IAM provisioning
            render_user_data.sh    user-data template substitution
            user-data.sh           VM first-boot script template
            trust-policy.json      EC2-assumes-role trust doc
            worker-policy.json     S3 R/W + EC2 self-terminate

---

## Quick start

### Local

    python -m pip install torch torchvision pyyaml wandb tensorboard

    # smoke test (CPU, ~5 min):
    bash scripts/dryrun.sh adamw

    # one full CIFAR-100 ResNet-18 run on a local GPU:
    python main.py \
        $(python scripts/infra/yaml_to_args.py configs/cifar100/resnet18.yaml) \
        $(bash scripts/infra/optim_flags.sh adame-ema-a999) \
        --output_dir runs/r18-adame-a999

### Local job-queue (no cloud)

    # uses local filesystem 'lease layer' if you point CKPT_BUCKET at a
    # local mount via aws-cli or just skip lease (single worker):
    bash scripts/launch_local.sh 1

### dstack (managed cloud)

    cp .env.example .env             # fill in WANDB_API_KEY, CKPT_BUCKET, …
    source .env
    bash scripts/infra/setup_aws.sh  # idempotent: creates bucket + IAM
    bash scripts/launch_workers.sh 2

### Bare-EC2 spot fleet (DyF/NELU-style)

    cp .env.example .env             # plus AMI, SG, SUBNET_<az>
    source .env
    bash scripts/infra/setup_aws.sh
    bash scripts/launch_campaign.sh  # watchdog keeps TARGET_WORKERS up

---

## Adding a new optimizer variant

One case-arm in `scripts/infra/optim_flags.sh` and one or more lines in
`scripts/infra/default_job_order.txt`. The orchestrator picks them up
automatically:

    optim_flags() {
        case "$key" in
            …
            adame-ema-a999-l30)
                printf '%s' "--optimizer adame --anchor ema \\
                    --ema_decay 0.999 --weight_decay 0.30"
                ;;
        esac
    }

---

## Output layout (per experiment)

`<exp>` = `<config-basename>-<variant>` (e.g. `resnet18-adame-ema-a999`).

    s3://$CKPT_BUCKET/<exp>/
        log.txt                 — append-only training log
        args.json               — full argparse snapshot
        checkpoint-<N>.pth      — rolling, capped at save_ckpt_num
        complete                — sentinel; presence ⇒ done
        lease                   — owner + ts; consumed by orchestrate.sh

The `complete` sentinel makes the watchdog idempotent: re-running the
campaign skips experiments that already have it.
