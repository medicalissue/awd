#!/usr/bin/env bash
# Launch ONE bare-EC2 spot worker for the AWD campaign.
#
# Usage:
#   source .env
#   bash scripts/infra/run_worker.sh <az> [<instance-type>] [<name-suffix>]
#
# Positional:
#   az              us-west-2a|b|c|d. Picks the matching subnet.
#   instance-type   Defaults to $INSTANCE_TYPE (.env), then g5.xlarge.
#                   CIFAR-100 fits comfortably on a single A10G/L4.
#   name-suffix     Tag suffix; defaults to current UTC timestamp.
#
# Required env (.env):
#   AMI                 AMI ID (Deep Learning Base AMI for your region)
#   SG                  security group id
#   IAM_PROFILE         awd-worker-profile
#   SUBNET_<AZ>         subnet id per AZ (e.g. SUBNET_us_west_2d=subnet-…)
#   REPO_URL, REPO_REF, CKPT_BUCKET, WANDB_*  — for render_user_data.sh
# Optional:
#   KEY                 EC2 key pair name (omit for IMDS-only access)
#   DATA_SNAPSHOT       snap-... (only if you mirror datasets in EBS)

set -euo pipefail

AZ="${1:?az required (e.g. us-west-2d)}"
INSTANCE_TYPE="${2:-${INSTANCE_TYPE:-g5.12xlarge}}"
NAME_SUFFIX="${3:-$(date -u +%Y%m%dT%H%M%S)}"

: "${AMI:?AMI required (Deep Learning Base AMI ID for your region)}"
: "${SG:?SG required (security group id)}"
: "${IAM_PROFILE:=awd-worker-profile}"
: "${REGION:=${AWS_DEFAULT_REGION:-us-west-2}}"

# Subnet lookup: SUBNET_us_west_2d, …
subnet_var="SUBNET_$(echo "$AZ" | tr '-' '_')"
SUBNET="${!subnet_var:-}"
[[ -z "$SUBNET" ]] && {
    echo "ERROR: $subnet_var not set in .env (subnet for $AZ)" >&2; exit 2;
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA_B64=$(bash "$SCRIPT_DIR/render_user_data.sh" | base64 | tr -d '\n')

NAME="awd-worker-${NAME_SUFFIX}"

echo "▶ launching ${NAME} in ${AZ} (${INSTANCE_TYPE})"
key_args=()
if [[ -n "${KEY:-}" ]]; then
    key_args=(--key-name "$KEY")
fi

# Block-device map: only a root volume by default. If the user wires
# DATA_SNAPSHOT, we attach a second volume from the snapshot. CIFAR is
# small enough that the default code path lets torchvision download.
bdm='[
    {"DeviceName":"/dev/sda1",
     "Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}'
if [[ -n "${DATA_SNAPSHOT:-}" ]]; then
    bdm+=',
    {"DeviceName":"/dev/sdg",
     "Ebs":{"SnapshotId":"'"$DATA_SNAPSHOT"'","VolumeSize":50,"VolumeType":"gp3",
            "DeleteOnTermination":true}}'
fi
bdm+=']'

aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    "${key_args[@]}" \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings "$bdm" \
    --user-data "$USER_DATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$NAME},
        {Key=Project,Value=awd},
        {Key=Role,Value=worker},
        {Key=Campaign,Value=${CAMPAIGN:-awd}}
    ]" \
    --query 'Instances[0].{InstanceId:InstanceId,AZ:Placement.AvailabilityZone,State:State.Name}' \
    --output json
