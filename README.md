# Fleet Computer Vision Lifecycle with SageMaker SDK v3 and Neo

Train a camera-specific anomaly detector and defect classifier, evaluate both
models, compile passing models with SageMaker Neo, and register their ONNX
artifacts in SageMaker Model Registry. The optional fleet extension queues
camera requests and limits concurrent pipeline executions.

The core lifecycle is:

```text
Preprocess ─┬─ Train autoencoder ─ Evaluate AUROC ─ Gate ─ Neo compile ─ Register
            └─ Train classifier ─ Evaluate recalls ─ Gate ─ Neo compile ─ Register
```

Each branch can warm-start from its camera's latest eligible model. Dataset
manifests record split identity and model ancestry. All processing and training
jobs use CPU instances; the default Neo target is Linux x86-64.

## Prerequisites

- A commercial AWS account and credentials with permission to deploy the
  resources in the CDK stacks. This sample targets `us-east-2`.
- AWS CLI v2, Git, Docker with Buildx, Node.js 22+, and AWS CDK CLI 2.1131.0+.
- Python 3.12 on Linux x86-64 or macOS arm64. The tested constraints do not
  support Intel macOS or Python 3.13+.
- SageMaker quotas sufficient for **3 simultaneous `ml.m5.xlarge` processing
  instances, 2 training instances, and 2 parallel compilation jobs** per
  pipeline execution. Check both instance-specific and aggregate quotas.
  A new account may require quota increases before its first run.

Before running AWS commands, select your profile and Region in the shell you
will use for this walkthrough. Replace `YOUR_PROFILE` with your AWS CLI profile
name, and confirm that the identity is the account you intend to use.
`AWS_PROFILE` also selects credentials for Python and the CDK application.

```bash
export AWS_PROFILE=YOUR_PROFILE
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION="$AWS_REGION"
aws sts get-caller-identity
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

Check quotas in that Region:

```bash
aws service-quotas list-service-quotas --service-code sagemaker --region "$AWS_REGION" \
  --query "Quotas[?contains(QuotaName,'ml.m5.xlarge for training job') ||
    contains(QuotaName,'ml.m5.xlarge for processing job') ||
    contains(QuotaName,'across all training jobs') ||
    contains(QuotaName,'across all processing jobs') ||
    contains(QuotaName,'parallel compilation')].[QuotaName,Value,QuotaCode]" \
  --output table
```

Use the returned quota codes to request increases through Service Quotas.
Fleet capacity multiplies these requirements by the configured concurrency.

## Deploy

Continue in the same shell with those environment variables set.

```bash
git clone https://github.com/aws-samples/sample-sagemaker-fleet-cv-pipeline.git
cd sample-sagemaker-fleet-cv-pipeline
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints-tested.txt

# Bootstrap CDK without synthesizing this application.
cdk bootstrap --app "" "aws://$ACCOUNT_ID/$AWS_REGION"

# First deployment: create the role, bucket, repository, and model groups.
cdk deploy FleetCVPipelineStack -c bootstrap=true --require-approval never

# Build linux/amd64 and save the digest from this push.
bash scripts/push_ecr_image.sh --digest-file .training-image-digest

# Add the complete pipeline using that exact image.
cdk deploy --all --require-approval never \
  -c "training_image_digest=$(cat .training-image-digest)"
```

CDK may briefly report a changeset waiting for review before continuing
automatically. Wait for the final deployment result. SDK notices about generated
job names and no jobs being started are expected while constructing the pipeline;
the run command below starts its jobs.

The first deployment has two phases because SageMaker SDK v3 validates the
execution role and uploads step code while constructing the pipeline definition.
Normal `cdk synth` therefore needs working AWS credentials, the deployed role and
bucket, and permissions including `iam:GetRole`, `iam:SimulatePrincipalPolicy`,
and S3 writes. Bootstrap synthesis skips definition construction.

Use `bootstrap=true` only for the initial prerequisites or destruction.
Deploying it over a complete stack removes the pipeline. When changing source,
rebuild the image and redeploy with its new digest. A digest fixes the deployed
image; rebuilding from the mutable base image can still produce a different digest.

## Run and verify

```bash
python scripts/run_pipeline.py
```

The helper starts one execution with synthetic data, waits for it, and verifies
the execution's own steps, both Approved model packages, completed Neo jobs,
and nonempty compiled artifacts in S3. It saves the evidence in `execution.json`.
Cloud startup can take several minutes per job; allow up to 90 minutes.
Wait for this command to finish successfully before continuing to inference or
the real-data example below.
If monitoring is interrupted, resume using the ARN printed at startup:

```bash
python scripts/run_pipeline.py --execution-arn YOUR_EXECUTION_ARN
```

The registered packages catalog ONNX models, metrics, and camera identity.
They reference the training container, which has no inference server, so they
are not ready to deploy to a SageMaker endpoint. Try local ONNX inference:

```bash
python scripts/inference_demo.py --model-group fleet-cv-anomaly-model-group --type anomalib
python scripts/inference_demo.py --model-group fleet-cv-classifier-model-group --type classifier
```

For downstream edge deployment, use the Neo artifact URIs in `execution.json`.
Linux x86-64 is the validation target; other devices require their own compile
and device-runtime validation.

## Use real data

The [BCCD dataset](https://github.com/Shenggan/BCCD_Dataset) provides an
illustrative image dataset. The downloader maps platelet presence to the defect
class; this is a teaching example, not a medical application.

```bash
python scripts/download_bccd.py --output-dir ./data/bccd --upload-s3
DATA_URI=$(python -c 'import json; print(json.load(open("data/bccd/publication_index.json"))["cameras"]["default"]["s3_uri"])')
python scripts/run_pipeline.py --data-s3-uri "$DATA_URI" --output execution-bccd.json
```

Pass `--upstream-ref COMMIT_SHA` to the downloader to pin the dataset revision.
It publishes content-addressed datasets with a completion marker. The URI must
point above the `anomalib/` format directory, within this sample's bucket.
The publication index supplies that exact URI.

The default `MinAUROC=0.55` is a demonstration floor. Historical BCCD validation
measured 0.6127 over 52 test images (18 good, 34 defect); this is a small-sample
point estimate, with no confidence interval. Choose thresholds using your own
data before using the pattern in an application. Warm-start parents and upstream
dataset changes can change subsequent results.

Split assignment uses decoded image content so metadata-only changes do not move
an image between splits. This protects against exact decoded-image duplication;
it does not detect near-duplicates, related frames, or patient/production-lot
overlap. Partition those groups before publishing real datasets.

The notebook in `notebooks/` provides an interactive data and inference walkthrough.
For multiple cameras and concurrency control, continue with [fleet scheduling](docs/fleet.md).

## Pipeline parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `TrainingInstanceType` / `ProcessingInstanceType` | `ml.m5.xlarge` | CPU job instances |
| `NumSamples` | `400` | Synthetic training sample count |
| `MinAUROC` | `0.55` | Anomaly AUROC gate |
| `MinBalancedAccuracy` | `0.6` | Classifier mean per-class recall gate |
| `MinDefectRecall` | `0.5` | Classifier defect recall gate |
| `CameraId` | `default` | Registry identity and warm-start selection |
| `TrainingDataS3Uri` | `none` | Published dataset prefix; otherwise generate synthetic data |
| `NeoTargetDevice` | `none` | Named device, overriding platform settings |
| `NeoTargetOs` / `NeoTargetArch` | `LINUX` / `X86_64` | Compilation platform |

Both classifier gates must pass. A failing branch neither compiles nor registers.
Parameters can also be passed with AWS CLI `start-pipeline-execution`.

## Test locally

```bash
pip install -r requirements-dev.txt -c constraints-tested.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/ -m "not integration" -q
python scripts/smoke_local.py
```

The smoke runs the real preprocessing, training, export, and evaluation scripts.
Its small default dataset tests handoffs and gate decisions. To require passing
gates, use `--num-samples 200 --epochs 6 --expect-gate-pass`.

After deploying prerequisites, check the generated pipeline and both stack modes:

```bash
python -m pytest tests/ -m integration -q
cdk synth -c bootstrap=true -c fleet=true
cdk synth -c fleet=true \
  -c "training_image_digest=$(cat .training-image-digest)"
```

Local tests cover metrics, split and lineage integrity, exported graphs, stack
structure, and selected fleet failure paths. A cloud execution is needed to
validate IAM, managed jobs, Neo compilation, and registration together.

## Destroy

Stop submitting work and let active executions finish. Cleanup permanently
deletes registered model versions; stack destruction deletes the artifact bucket
(including object versions and datasets) and every image in the training
repository. Copy any results you want to retain first.

Use the same environment and activated virtual environment as deployment:

```bash
python scripts/cleanup.py --dry-run
python scripts/cleanup.py && \
  cdk destroy --all -c bootstrap=true -c fleet=true --force
python scripts/cleanup.py --verify-deleted
```

This works with the core stack alone or with fleet scheduling enabled. For a
fleet deployment, cleanup first closes admission and disables queue consumption;
it refuses to delete while leases, queued work, pipeline executions, or Neo jobs
remain. See [fleet recovery](docs/fleet.md#recovery-and-cleanup) if it reports
retained work. Admission remains closed until explicitly reopened.

Verification checks that both sample stacks, the S3 bucket, ECR repository, and
model groups are absent. The shared `CDKToolkit` bootstrap resources and
SageMaker job history/logs remain. No endpoint or other always-on compute is
created by this sample.

## Cost and scope

Charges come from processing/training time, Neo compilation, storage, and optional
fleet service requests. Use current Regional prices and your execution's job
durations to estimate cost; startup and minimum billing can matter more than
training time for small examples. There are no always-on instances by default.

This is a deployable sample for commercial AWS Regions. Review the IAM policies
before adapting it: ECR pulls and data access are scoped to sample resources,
while some service actions require `"*"` resources. The execution role includes
EC2 network-interface permissions for SageMaker's role validation.

See [CONTRIBUTING](CONTRIBUTING.md) for security issue reporting.
This project is licensed under [MIT-0](LICENSE).
