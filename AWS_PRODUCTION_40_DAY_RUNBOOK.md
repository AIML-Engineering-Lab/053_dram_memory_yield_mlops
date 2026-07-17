# P053 AWS Production 40-Day Runbook

> Purpose: one tracked, step-by-step guide to run the full enterprise AWS production simulation after EC2 GPU quota approval.
> Scope: Day 1 AWS GPU training, v1 champion registration, automated 40-day data generation, drift detection, retraining, canary promotion or rollback, S3 cost cleanup, and final AWS shutdown.
> Date started: 2026-05-24

## Final Status — 2026-07-11

The intended live AWS daily production run is complete. A 2026-07-17 audit found the schedule kept running through unintended Days 41-45. S3 `state/pipeline_state.json` now reports `current_day=46`, `last_completed_day=45`, `status=complete`, and `last_run=2026-07-16T02:05:57Z`. The active champion is `s3://p053-mlflow-artifacts/models/day30_v2_retrained.pt` with `champion_updated_day=30`.

AWS compute shutdown has been verified: EC2 `i-0562654a22d44346f` is stopped, RDS `p053-mlflow-db` is stopped, and no NAT gateways are available or pending. This stops EC2/RDS/NAT compute charges, but it does not guarantee an exact `$0` future bill. Remaining billable inventory from the final audit: 125 GiB gp3 EBS, 20 GiB RDS storage, 21 automated RDS snapshots, ~4.80 GB current S3 data with 387 object versions, 1 ECR repo, and 1 associated public IPv4/EIP. Retention decision: keep minimal evidence resources for now and accept small residual charges.

Final artifact evidence is committed in `docs/S3_Artifacts_Inventory_Report.html` and `docs/S3_Artifacts_Inventory_Report.pdf`. That report captures S3 current objects, version counts, day-wise artifacts, model artifacts, largest objects, Day 41-45 accidental extras, and the retained AWS cost posture.

## Success Criteria

- [x] EC2 `g4dn.xlarge` launched and supported the production daily run path.
- [x] RDS `p053-mlflow-db` was used only for the production window and is now stopped.
- [x] MLflow/S3 artifact state preserved champion lineage and production artifacts.
- [x] Day 1 A100 champion was uploaded to S3 and used as v1 starting point.
- [x] GitHub Actions + Airflow daily DAG completed days 1-40.
- [x] Daily synthetic production Parquet data was generated automatically with variable volumes.
- [x] Low-data or unreliable drift is tagged but does not trigger retraining.
- [x] Drift threshold/staleness triggered v2 promotion after canary on Day 30.
- [x] Day 39 bad deploy or canary failure kept/rolled back to the v2 champion.
- [ ] S3 raw production data is deleted or expired on a 10-day cost policy while models and reports are retained.
- [x] End-of-run cleanup stops EC2 and RDS, deletes/avoids any p053 NAT gateway, and verifies no expensive compute resources remain running.

## Automation Map

| Layer | File or AWS Service | Role |
|---|---|---|
| EC2 bootstrap | `deploy/aws/ec2-user-data.sh` | Installs Docker, NVIDIA drivers, Python deps, builds Airflow GPU image, starts stack when env exists. |
| Production stack | `deploy/aws/docker-compose-bigdata-aws.yml` | Kafka, Spark, Airflow, MLflow to RDS, S3 artifacts, Prometheus, Grafana. |
| Master orchestration | `deploy/airflow/dags/dag_simulation_master.py` | Triggers all 40 simulated days and calls auto-stop for `phase3`. |
| Daily pipeline | `deploy/airflow/dags/dag_daily_yield_pipeline.py` | Generates data, publishes Kafka, runs Spark ETL, detects drift, uploads artifacts to S3. |
| Retraining | `deploy/airflow/dags/dag_retrain_pipeline.py` | Runs real `src.train` GPU training, canary evaluation, MLflow promote or rollback. |
| Variable data | `src/streaming_data_generator.py` | `phase2`: 2M-9M rows/day. `phase3`: 30M-350M rows/day. |
| S3 artifacts | `src/s3_utils.py` | Uploads `models/`, `data/production/`, `drift/`, `retrain/`, `canary/`. |
| Cost shutdown | `src/ec2_auto_stop.py` | Stops RDS, deletes p053 NAT gateway, stops EC2 at phase3 completion. |

## Production Storyline To Preserve

| Day(s) | Scenario | Expected Behavior |
|---|---|---|
| Day 1 | Initial full AWS training | Train on full 16M dataset or approved huge Parquet source, log to MLflow, save v1 champion to S3. |
| Days 1-8 | Steady state | Establish reference window, no retrain. |
| Day 9 | False alarm | Single feature spike, log warning only. |
| Day 10 | Auto recover | Spike disappears, no retrain. |
| Days 11-18 | Gradual drift | Drift warnings accumulate, no retrain before staleness gate. |
| Day 19 | Sudden shift | New probe card appears, stronger drift signal. |
| Day 20 | Threshold breach 1 | Drift threshold can be crossed, but retrain is blocked by staleness gate. |
| Day 26 | Threshold breach 2 | More drift, still blocked by staleness gate. |
| Day 30 | Retrain trigger | Drift/staleness gate opened; v2 retrain promoted to champion and uploaded to S3. |
| Day 31 | v2 champion validation | Day 31+ uses the Day 30 retrained model; Spark/disk recovery issues were fixed. |
| Days 32-35 | Post-retrain recovery | v2 champion handles new distribution; cooldown gate prevents immediate retrain loops. |
| Days 36-38 | Second drift | New recipe distribution appears. |
| Day 39 | Bad deploy or canary failure | Canary fails deliberately or by metric degradation, rollback restores previous champion. |
| Day 40 | Final recovery | Final run completes, auto-stop cleanup executes for `phase3`. |

## Cost Guardrails

- EC2 `g4dn.xlarge`: about `$0.526/hr`; stop immediately after run or review.
- RDS `db.t3.micro`: about `$0.018/hr`; stop after run, but remember AWS auto-restarts stopped RDS after 7 days.
- S3 Standard: about `$0.023/GB/month`; raw daily Parquet must be lifecycle-managed or deleted.
- NAT Gateway: about `$0.045/hr` plus data processing; delete any p053 NAT gateway after the run.
- Billing alarm target: `$200` USD through `src.ec2_auto_stop --setup-alarm --threshold 200`.
- Never delete these S3 prefixes during cleanup: `models/`, `mlartifacts/`, `benchmarks/`, `drift/`, `retrain/`, `canary/`, `reports/`.

### Public IPv4 and VPC Cost Note

- If RDS is `PubliclyAccessible=true`, AWS can bill a public IPv4 charge under `Amazon Virtual Private Cloud` at about `$0.005/hr`.
- That VPC charge is separate from the RDS compute line item.
- It is not caused by the EC2 GPU quota request.
- It is not caused by EC2 when no EC2 instance is running.
- For the 40-day production execution, a public RDS endpoint is not strictly required if EC2 and RDS communicate privately inside the same VPC.
- The current project can run with the existing public RDS for simplicity, but that may continue the VPC public IPv4 charge whenever AWS keeps the address allocated.
- Lowest-cost approach between runs: stop RDS immediately after the run. Lower-cost networking approach during production: use private RDS instead of public RDS.

## Billing Safety Prerequisite: Fix IAM Before Production

Current verified IAM gap for `p053-cicd-user`:

- `cloudwatch:PutMetricAlarm` = allowed
- `sns:CreateTopic` = denied
- `ce:GetCostAndUsage` = denied
- `servicequotas:ListRequestedServiceQuotaChangeHistoryByQuota` = denied

You need a more privileged AWS identity once to fix this.

### Option A: Grant The Missing Permissions To `p053-cicd-user`

1. Sign in to AWS as root or an admin IAM user.
2. Open IAM.
3. Go to Users.
4. Open `p053-cicd-user`.
5. Open Permissions.
6. Click Add permissions.
7. Click Create inline policy.
8. Open the JSON tab.
9. Paste this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "P053BillingSafety",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricAlarm",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:DeleteAlarms",
        "sns:CreateTopic",
        "sns:GetTopicAttributes",
        "sns:SetTopicAttributes",
        "sns:Subscribe",
        "sns:ListTopics",
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
        "servicequotas:GetServiceQuota",
        "servicequotas:ListServiceQuotas",
        "servicequotas:ListRequestedServiceQuotaChangeHistoryByQuota"
      ],
      "Resource": "*"
    }
  ]
}
```

10. Click Next.
11. Name the policy `p053-billing-safety-inline`.
12. Click Create policy.
13. Re-run these commands locally:

```bash
set -a && source deploy/aws/.env.aws && set +a
./.venv/bin/python -m src.ec2_auto_stop --setup-alarm --threshold 200

aws ce get-cost-and-usage \
  --region us-east-1 \
  --time-period Start=2026-05-01,End=2026-06-01 \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

### Option B: Create The Billing Alarm Manually In AWS Console

Use this if you do not want to change IAM right now.

1. Sign in as root or an admin IAM user.
2. Open Billing and Cost Management.
3. Open Billing preferences.
4. Enable `Receive CloudWatch Billing Alerts`.
5. Switch region to `us-east-1`.
6. Open SNS.
7. Create a Standard topic named `p053-billing-alarm`.
8. Add an email subscription for yourself.
9. Confirm the email subscription.
10. Open CloudWatch.
11. Open Alarms.
12. Click Create alarm.
13. Select metric.
14. Choose `Billing`.
15. Choose `Total Estimated Charge`.
16. Choose `USD`.
17. Set threshold type to Static.
18. Set the alarm to trigger when charge is greater than `200`.
19. Choose the SNS topic `p053-billing-alarm`.
20. Name the alarm `p053-budget-safety-stop`.
21. Create the alarm.

### Recommended Check After Either Option

Local machine note:

- On your Mac, use `deploy/aws/.env.aws`.
- `deploy/aws/.env` is created later on the EC2 instance in Phase 5.

```bash
aws cloudwatch describe-alarms \
  --region us-east-1 \
  --alarm-types MetricAlarm \
  --query 'MetricAlarms[*].{AlarmName:AlarmName,State:StateValue,Threshold:Threshold,Metric:MetricName}'
```

## Phase 0: Local Preflight

- [ ] Activate the local environment.

```bash
source .venv/bin/activate
```

- [ ] Confirm AWS identity and region.

```bash
aws sts get-caller-identity
aws configure get region
```

- [ ] Confirm GPU quota is usable for one `g4dn.xlarge`.

```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region us-west-2
```

- [ ] Check all live AWS services before starting anything billable.

```bash
bash check_aws_services.sh
```

- [ ] Confirm S3 bucket and ECR repo exist.

```bash
aws s3api head-bucket --bucket p053-mlflow-artifacts
aws ecr describe-repositories \
  --repository-names 053-memory-yield-predictor \
  --region us-west-2
```

- [ ] Confirm RDS exists. It may be `stopped` before launch.

```bash
aws rds describe-db-instances \
  --db-instance-identifier p053-mlflow-db \
  --region us-west-2 \
  --query 'DBInstances[0].{id:DBInstanceIdentifier,status:DBInstanceStatus,endpoint:Endpoint.Address}'
```

## Phase 1: Start RDS And Network Access

- [ ] Start RDS if it is stopped.

```bash
aws rds start-db-instance \
  --db-instance-identifier p053-mlflow-db \
  --region us-west-2

aws rds wait db-instance-available \
  --db-instance-identifier p053-mlflow-db \
  --region us-west-2
```

- [ ] Get the current laptop public IP.

```bash
MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '\n')
echo "$MY_IP"
```

- [ ] Authorize required EC2 UI ports for the current IP. Ignore duplicate-rule errors.

```bash
SG_ID=sg-0f11ba29c1155cba3
for PORT in 22 3000 5001 8000 8080 8888 9000 9090; do
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port "$PORT" \
    --cidr "$MY_IP/32" \
    --region us-west-2 || true
done
```

Port map:

| Port | Service |
|---|---|
| 22 | SSH |
| 3000 | Grafana |
| 5001 | MLflow |
| 8000 | FastAPI if started separately |
| 8080 | Spark UI |
| 8888 | Airflow UI for `docker-compose-bigdata-aws.yml` |
| 9000 | Kafdrop |
| 9090 | Prometheus |

## Phase 2: Create Or Verify EC2 IAM Role

The production compose file expects AWS access through the EC2 instance role instead of hardcoded keys.

Access note:

- `p053-cicd-user` does not currently have enough IAM rights for this phase by default.
- Verified denied actions: `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:CreateInstanceProfile`, `iam:AddRoleToInstanceProfile`, `iam:PassRole`.
- If you hit `AccessDenied` here, that is expected with the current user.
- You must do one of these before continuing:
  - use a root or admin AWS identity once for Phase 2 and the EC2 launch in Phase 4, or
  - attach the inline policy below to `p053-cicd-user`, then continue with the same CLI commands.

### Option A: Use Root Or Admin Once For Phase 2

1. Sign in as root or an admin IAM user.
2. Attach the EC2 role resources using the commands in this phase.
3. If you do not grant `iam:PassRole` to `p053-cicd-user`, also perform the EC2 launch in Phase 4 as that same admin identity.

### Option B: Grant Minimal EC2 Role Bootstrap Permissions To `p053-cicd-user`

1. Sign in as root or an admin IAM user.
2. Open IAM.
3. Open Users.
4. Open `p053-cicd-user`.
5. Open Permissions.
6. Click Add permissions.
7. Click Create inline policy.
8. Open the JSON tab.
9. Paste this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "P053RoleCreateAndAttach",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:GetRole",
        "iam:AttachRolePolicy"
      ],
      "Resource": "arn:aws:iam::718036735422:role/p053-ec2-role"
    },
    {
      "Sid": "P053InstanceProfileManage",
      "Effect": "Allow",
      "Action": [
        "iam:CreateInstanceProfile",
        "iam:GetInstanceProfile",
        "iam:AddRoleToInstanceProfile"
      ],
      "Resource": "arn:aws:iam::718036735422:instance-profile/p053-ec2-profile"
    },
    {
      "Sid": "P053PassRoleToEc2",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::718036735422:role/p053-ec2-role",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "ec2.amazonaws.com"
        }
      }
    }
  ]
}
```

10. Click Next.
11. Name the policy `p053-ec2-role-bootstrap-inline`.
12. Click Create policy.
13. Re-run the commands in Phase 2 from your current terminal.

- [ ] Create an EC2 trust policy if the role does not already exist.

```bash
cat > /tmp/p053-ec2-trust-policy.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON

aws iam create-role \
  --role-name p053-ec2-role \
  --assume-role-policy-document file:///tmp/p053-ec2-trust-policy.json || true
```

- [ ] Attach the minimum project policies for this run.

```bash
aws iam attach-role-policy \
  --role-name p053-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name p053-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

aws iam attach-role-policy \
  --role-name p053-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess

aws iam attach-role-policy \
  --role-name p053-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonRDSFullAccess
```

- [ ] Create or reuse an instance profile.

```bash
aws iam create-instance-profile \
  --instance-profile-name p053-ec2-profile || true

aws iam add-role-to-instance-profile \
  --instance-profile-name p053-ec2-profile \
  --role-name p053-ec2-role || true
```

## Phase 3: Configure S3 Lifecycle Cleanup Before Data Starts

Because S3 bucket versioning is enabled, deleting current objects is not enough. This lifecycle rule expires current and noncurrent raw production data while retaining models and MLflow artifacts.

- [ ] Apply 10-day raw production data lifecycle.

```bash
cat > /tmp/p053-s3-lifecycle.json <<'JSON'
{
  "Rules": [
    {
      "ID": "expire-raw-production-parquet-after-10-days",
      "Status": "Enabled",
      "Filter": {"Prefix": "data/production/"},
      "Expiration": {"Days": 10},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 10},
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}
    }
  ]
}
JSON

aws s3api put-bucket-lifecycle-configuration \
  --bucket p053-mlflow-artifacts \
  --lifecycle-configuration file:///tmp/p053-s3-lifecycle.json
```

- [ ] Verify lifecycle is active.

```bash
aws s3api get-bucket-lifecycle-configuration \
  --bucket p053-mlflow-artifacts
```

## Phase 4: Launch EC2 `g4dn.xlarge`

Launch note:

- If `p053-cicd-user` does not have `iam:PassRole` on `p053-ec2-role`, this phase will fail even if the role already exists.
- If you used Phase 2 Option B above, you can continue with this phase from your normal terminal.
- If you used Phase 2 Option A without granting `iam:PassRole`, do the EC2 launch as root or admin.

- [ ] Launch the GPU instance with 125 GB gp3 root storage and auto-bootstrap.

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0e1c254c6491f13f2 \
  --instance-type g4dn.xlarge \
  --key-name p053-key \
  --security-group-ids sg-0f11ba29c1155cba3 \
  --iam-instance-profile Name=p053-ec2-profile \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=125,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data file://deploy/aws/ec2-user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=p053-aws-production},{Key=Project,Value=p053}]' \
  --region us-west-2 \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region us-west-2
```

- [ ] Capture public IP and update local tracking.

```bash
EC2_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region us-west-2 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "$EC2_IP"
```

- [ ] SSH and watch bootstrap.

```bash
ssh -i ~/.ssh/p053-key.pem ec2-user@$EC2_IP
sudo tail -f /var/log/p053-bootstrap.log
```

Bootstrap is complete when `/home/ec2-user/.p053_bootstrap_status` exists.

## Phase 5: Copy AWS Environment And Start Stack

The repo intentionally does not commit live `.env` files. Copy the local live AWS env to EC2 as `deploy/aws/.env`.

- [ ] Copy the local env file to EC2.

```bash
scp -i ~/.ssh/p053-key.pem \
  deploy/aws/.env.aws \
  ec2-user@$EC2_IP:/home/ec2-user/053_memory_yield_predictor/deploy/aws/.env
```

- [ ] SSH into EC2 and verify GPU.

```bash
ssh -i ~/.ssh/p053-key.pem ec2-user@$EC2_IP
nvidia-smi
```

- [ ] Start or restart the production big data stack.

```bash
cd /home/ec2-user/053_memory_yield_predictor
docker compose -f deploy/aws/docker-compose-bigdata-aws.yml \
  --env-file deploy/aws/.env \
  up -d
```

- [ ] Verify services.

```bash
docker ps
curl -f http://localhost:8888/health
curl -f http://localhost:5001/health
curl -f http://localhost:9090/-/healthy
```

Browser URLs:

```text
Airflow:    http://<EC2_IP>:8888  admin/admin
MLflow:     http://<EC2_IP>:5001
Spark UI:   http://<EC2_IP>:8080
Grafana:    http://<EC2_IP>:3000
Kafdrop:    http://<EC2_IP>:9000
Prometheus: http://<EC2_IP>:9090
```

## Phase 6: Set Billing Alarm And Shutdown Safety Net

Environment file note:

- On EC2, use `deploy/aws/.env` because Phase 5 copies it into the instance.
- On your local machine, use `deploy/aws/.env.aws`.

- [ ] Create or refresh the `$200` billing alarm.

```bash
cd /home/ec2-user/053_memory_yield_predictor
set -a && source deploy/aws/.env && set +a
python -m src.ec2_auto_stop --setup-alarm --threshold 200
```

- [ ] Dry-run cleanup before production.

```bash
python -m src.ec2_auto_stop --dry-run
python -m src.ec2_auto_stop --stop-rds --dry-run
python -m src.ec2_auto_stop --delete-nat --dry-run
```

## Phase 7: Day 1 AWS GPU Training To Create v1 Champion

Use this when we want the story to be fully AWS-native, even though the earlier A100 champion exists on S3. The training script consumes the full preprocessed training matrix `data/preprocessed_full.npz`, which is DVC-tracked on the project S3 remote and was built from the large production-style DRAM dataset.

- [ ] Ensure the full training dataset is present on EC2.

```bash
cd /home/ec2-user/053_memory_yield_predictor
ls -lh data/preprocessed_full.npz || dvc pull data/preprocessed_full.npz.dvc

# Fallback only if DVC is unavailable on the instance:
aws s3 ls s3://p053-mlflow-artifacts/dvc/ --recursive | grep b07e41d3f53e98c7f169b34a770f4dcd
```

- [ ] Run full Day 1 training on EC2 T4.

```bash
cd /home/ec2-user/053_memory_yield_predictor
set -a && source deploy/aws/.env && set +a

python -m src.train \
  --full \
  --epochs 50 \
  --batch-size 4096 \
  --run-name day1-aws-v1-champion \
  --context aws-day1
```

- [ ] Upload v1 model and benchmark to S3.

```bash
aws s3 cp src/artifacts/ s3://p053-mlflow-artifacts/models/day01_v1_champion/ \
  --recursive \
  --exclude '*' \
  --include '*.pt'

aws s3 cp data/benchmark_*.json s3://p053-mlflow-artifacts/benchmarks/day01/ || true
```

- [ ] Verify v1 in MLflow and S3.

```bash
aws s3 ls s3://p053-mlflow-artifacts/models/day01_v1_champion/ --recursive
aws s3 ls s3://p053-mlflow-artifacts/benchmarks/day01/ --recursive
```

Manual verification: open MLflow and confirm the `day1-aws-v1-champion` run exists with GPU `Tesla T4`, metrics, and artifacts.

## Phase 8: Airflow Preflight Automation Test

- [ ] Unpause DAGs.

```bash
docker exec p053-airflow-web airflow dags unpause p053_daily_yield_pipeline
docker exec p053-airflow-web airflow dags unpause p053_retrain_pipeline
docker exec p053-airflow-web airflow dags unpause p053_simulation_master
```

- [ ] Run one small daily pipeline test.

```bash
docker exec p053-airflow-web airflow dags trigger p053_daily_yield_pipeline \
  --conf '{"day_number": 1, "n_rows": 100000}'
```

- [ ] Verify day 1 test artifacts locally and in S3.

```bash
ls -lh data/production/day_01.parquet data/drift_reports/ || true
aws s3 ls s3://p053-mlflow-artifacts/data/production/day_01/ --recursive || true
```

Important preflight note: if the daily DAG uses Airflow `params` defaults instead of `dag_run.conf`, confirm in the Airflow UI that the triggered run used the requested day and row count. If it ignored the JSON conf, patch the DAG before the full run.

## Phase 9: Full Automated 40-Day Run

Choose the production scale intentionally:

| Scale | Daily Volume | Use Case |
|---|---:|---|
| `phase2` | 2M-9M rows/day | Accelerated demo, lower cost, faster validation. |
| `phase3` | 30M-350M rows/day | Full enterprise stress run with very huge days. |

- [ ] Export production env for phase3.

```bash
cd /home/ec2-user/053_memory_yield_predictor
set -a && source deploy/aws/.env && set +a
export SIMULATION_SCALE=phase3
export MODEL_PARAMS=317000
export EC2_INSTANCE_TYPE=g4dn.xlarge
```

- [ ] Trigger the master DAG once. It runs all 40 days.

```bash
docker exec p053-airflow-web airflow dags trigger p053_simulation_master \
  --conf '{"scale": "phase3"}'
```

- [ ] Monitor from CLI.

```bash
docker exec p053-airflow-web airflow dags list-runs -d p053_simulation_master
docker logs -f p053-airflow-scheduler
```

- [ ] Monitor from browser.

```text
Airflow: http://<EC2_IP>:8888
MLflow:  http://<EC2_IP>:5001
Grafana: http://<EC2_IP>:3000
S3:      s3://p053-mlflow-artifacts/
```

Actual completed events:

- Days 1-8: generate steady data and reference window.
- Days 9-30: drift warnings, threshold breaches, no retrain until staleness gate passes.
- Day 30: retrain/canary promoted `day30_v2_retrained.pt` as champion.
- Day 31-33: production recovery fixed Spark memory, EC2 disk saturation, accidental retrain waits, and Kafka JVM heap risk.
- Day 34-40: intended scheduled GitHub Actions runs completed successfully.
- Day 39: canary failure or deliberate rollback path verified; v2 champion retained.
- Day 40: final intended recovery completed; state advanced to `current_day=41`, `status=complete`; EC2/RDS stopped and NAT none.
- Day 41-45: unintended scheduled GitHub Actions runs still executed because the completion check exited only one step, not the job.
- 2026-07-17: workflow cron disabled and AWS-starting steps guarded by `should_run` so complete/day>40 state cannot start RDS/EC2.

## Phase 10: S3 Cleanup During And After Run

The lifecycle rule is the automatic long-term guard. For immediate project-cost control after verifying screenshots and reports, manually delete raw Parquet prefixes only.

- [ ] Check S3 size.

```bash
aws s3api list-objects-v2 \
  --bucket p053-mlflow-artifacts \
  --query 'sum(Contents[].Size)'
```

- [ ] Delete one very huge raw-data day after verification. Replace `XX` with day number.

```bash
aws s3 rm s3://p053-mlflow-artifacts/data/production/day_XX/ --recursive
```

- [ ] Delete all raw production Parquet after the report is complete.

```bash
aws s3 rm s3://p053-mlflow-artifacts/data/production/ --recursive
```

- [ ] Because versioning is enabled, ensure lifecycle handles noncurrent versions. Do not use ad hoc version deletion unless storage remains high after lifecycle processing.

```bash
aws s3api get-bucket-lifecycle-configuration --bucket p053-mlflow-artifacts
```

- [ ] Retain evidence artifacts.

```bash
aws s3 ls s3://p053-mlflow-artifacts/models/ --recursive
aws s3 ls s3://p053-mlflow-artifacts/mlartifacts/ --recursive --summarize
aws s3 ls s3://p053-mlflow-artifacts/drift/ --recursive
aws s3 ls s3://p053-mlflow-artifacts/retrain/ --recursive
aws s3 ls s3://p053-mlflow-artifacts/canary/ --recursive
```

## Phase 11: End-Of-Run Evidence Capture

- [ ] MLflow screenshots: v1 champion, v2 retrain, rollback run, model registry alias.
- [ ] Airflow screenshots: master DAG success, day 31 retrain DAG, day 39 rollback DAG.
- [ ] Grafana screenshots: system metrics, throughput, service health.
- [ ] S3 screenshots: retained model/artifact prefixes and raw-data lifecycle rule.
- [ ] AWS screenshots: EC2 stopped, RDS stopped, CloudWatch alarm, final cost.
- [ ] Export key artifacts locally if needed for report generation.
  - CLI audit on 2026-07-11 verified EC2 stopped, RDS stopped, NAT none, and Day 40 S3/GitHub completion. Browser screenshots remain optional evidence.

```bash
aws s3 sync s3://p053-mlflow-artifacts/benchmarks/ data/aws_benchmarks/
aws s3 sync s3://p053-mlflow-artifacts/drift/ data/aws_drift_reports/
aws s3 sync s3://p053-mlflow-artifacts/retrain/ data/aws_retrain_results/
aws s3 sync s3://p053-mlflow-artifacts/canary/ data/aws_canary_results/
```

## Phase 12: Final Cleanup And Stop Everything

Run this even if `ec2_auto_stop.py` already fired. Trust but verify.

- [ ] From EC2, attempt graceful project cleanup.

```bash
cd /home/ec2-user/053_memory_yield_predictor
set -a && source deploy/aws/.env && set +a
python -m src.ec2_auto_stop --phase phase3
```

- [ ] From local machine, stop EC2 if it is still running.

```bash
aws ec2 stop-instances \
  --instance-ids "$INSTANCE_ID" \
  --region us-west-2
```

- [ ] Stop RDS.

```bash
aws rds stop-db-instance \
  --db-instance-identifier p053-mlflow-db \
  --region us-west-2 || true
```

- [ ] Delete any p053 NAT gateway.

```bash
aws ec2 describe-nat-gateways \
  --region us-west-2 \
  --filter Name=tag:Project,Values=p053 \
  --query 'NatGateways[?State==`available`].NatGatewayId' \
  --output text

aws ec2 delete-nat-gateway \
  --nat-gateway-id <NAT_GATEWAY_ID> \
  --region us-west-2
```

- [ ] Verify no expensive resources remain running.

```bash
bash check_aws_services.sh
```

Expected final state:

| Resource | Final State |
|---|---|
| EC2 `p053-aws-production` | `stopped` or `terminated` |
| RDS `p053-mlflow-db` | `stopped` |
| NAT Gateway | deleted or none found |
| S3 raw production data | deleted manually or lifecycle-expiring after 10 days |
| S3 models and MLflow artifacts | retained |
| ECR repository | retained unless project is fully archived |

## Troubleshooting

| Symptom | Check | Fix |
|---|---|---|
| EC2 launch fails with quota error | `aws service-quotas get-service-quota ...` | Confirm quota approval is in `us-west-2` and covers G/VT On-Demand vCPUs. |
| `nvidia-smi` not found | `/var/log/p053-bootstrap.log` | Reboot once after driver install, then rerun NVIDIA toolkit steps if needed. |
| Stack does not start | `docker compose ... ps` | Ensure `deploy/aws/.env` exists on EC2 and RDS is `available`. |
| MLflow cannot connect | `curl localhost:5001/health`, RDS SG, password | Start RDS, verify DB endpoint/password, allow EC2 security group to reach RDS port 5432. |
| S3 uploads fail | `aws sts get-caller-identity` on EC2 | Attach or fix `p053-ec2-profile` IAM role. |
| Airflow trigger uses wrong day/rows | Airflow run conf vs task logs | Patch daily DAG to read `dag_run.conf` or pass values through Airflow params explicitly. |
| Raw S3 bill grows | `aws s3 ls --summarize`, lifecycle config | Delete `data/production/` prefixes after report capture and verify noncurrent expiration rule. |
| RDS restarts later | `check_aws_services.sh` weekly | AWS restarts stopped RDS after 7 days; stop it again or snapshot/delete after final archive. |

## Execution Log

Use this section as the live tracker.

| Date | Step | Result | Evidence Link or Note |
|---|---|---|---|
| 2026-05-24 | GPU quota approved | Ready to launch EC2 `g4dn.xlarge` | User confirmation |
| 2026-06-02 | Billing safety IAM fixed | `p053-billing-safety-inline` created successfully | User completed IAM inline policy |
| 2026-06-02 | Billing alarm created | `p053-budget-safety-stop` verified in CloudWatch | Threshold `$200`, SNS action attached |
| 2026-06-02 | Phase 0 preflight | Passed | AWS identity OK, region `us-west-2`, quota value `4`, S3 and ECR reachable |
| 2026-06-02 | RDS stopped | Complete | `p053-mlflow-db` status confirmed `stopped` |
| 2026-06-02 | Phase 2 complete | `p053-ec2-role` + `p053-ec2-profile` created and verified | `aws iam get-role` and `get-instance-profile` confirmed |
| 2026-06-02 | Phase 3 complete | S3 lifecycle rule active on `data/production/` prefix | 10-day expiry for current + noncurrent versions |
| 2026-07-01 | Day 30 retrain recovery | Fixed retrain wait/timeout path and ensured S3 state updates after canary promotion | `daily_pipeline.yml` timeout raised; retrain artifacts/state uploaded |
| 2026-07-03 | Day 31 recovery | Fixed Spark zombie/OOM and EC2 disk saturation | Spark memory reduced; Docker volumes/prune cleanup performed |
| 2026-07-04 | Day 32-33 recovery | Fixed accidental retrain wait and Kafka memory risk | champion cooldown from S3; `KAFKA_HEAP_OPTS` and restart policy committed |
| 2026-07-11 | Day 40 complete | Intended 40-day live AWS daily run complete | S3 state initially `current_day=41`, `last_completed_day=40`, `status=complete` |
| 2026-07-11 | EC2/RDS/NAT stopped | No expensive compute resources running | EC2 stopped, RDS stopped, NAT none; residual storage/IP cleanup decision remains |
| 2026-07-17 | Day 41-45 schedule leak fixed | Daily cron disabled and expensive steps guarded | S3 state observed `current_day=46`, `last_completed_day=45`; champion still Day 30 v2 |
