# Fleet-Scale Computer Vision Training Pipeline with SageMaker SDK v3

This sample deploys an end-to-end MLOps pipeline using the **SageMaker Python SDK v3** that trains, evaluates, and registers dual computer vision models (anomaly detection + defect classification) with ONNX export, automated quality gates, and fleet-scale orchestration via AWS Step Functions.

Use this as a companion to the AWS blog post: *Scaling Fleet-Wide Computer Vision Model Lifecycle Management at the Edge with Amazon SageMaker*.

## Architecture

![Solution architecture: SQS FIFO queue triggers Lambda, which starts a Step Functions state machine with a DynamoDB semaphore for concurrency control. The state machine launches a 13-step SageMaker Pipeline with parallel entry steps (Preprocess, Fetch Latest Anomaly, Fetch Latest Classifier), dual training branches with ONNX evaluation, quality gates, Neo compilation, and Model Registry registration. Compiled artifacts land in S3. Completions and failures converge on an SNS topic.](assets/architecture.png)

The solution consists of two CDK stacks:

1. **SageMakerCVPipelineStack** -- Defines a SageMaker Pipeline with dual training branches (a convolutional autoencoder for anomaly detection and a ResNet18 defect classifier), per-camera warm-start fetch steps for both branches, ONNX evaluation, quality gates, SageMaker Neo edge compilation, and model registration into the `sm1k-anomaly-model-group` and `sm1k-classifier-model-group` Model Registry groups. Registration depends on compilation, so only models that compile for the edge target become Approved packages.
2. **OrchestrationStack** -- Provides fleet-scale orchestration via SQS → Lambda → Step Functions with a DynamoDB-based semaphore for concurrency control (default: 5 concurrent pipeline executions). Executions that end in an operational failure state are forwarded to the completion SNS topic by an EventBridge rule, preserving the original error and cause.

## Prerequisites

- An AWS account with permissions to create SageMaker, Step Functions, Lambda, SQS, SNS, DynamoDB, ECR, and S3 resources
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with credentials
- [AWS CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/getting-started.html) version 2.1131.0 or later (`npm install -g aws-cdk`) -- older CLI versions fail with a cloud assembly schema version mismatch against the `aws-cdk-lib` this project installs
- Python 3.10–3.12 (Python 3.13+ is incompatible with the `numpy<2` constraint in `requirements.txt`; if your default `python3` is newer, create the virtual environment with an explicit interpreter, e.g. `python3.12 -m venv .venv`)
- Docker
- Configured AWS credentials are required for `cdk synth` and `cdk deploy` -- the stack resolves your account and region at synthesis time to build the pipeline definition
- To deploy to a region other than your AWS CLI default, export **all three** region variables before running any `cdk` or `aws sagemaker` command: `AWS_REGION`, `AWS_DEFAULT_REGION`, and `CDK_DEFAULT_REGION`. The CDK CLI reads `CDK_DEFAULT_REGION`, the SageMaker Python SDK uses boto3 which reads `AWS_DEFAULT_REGION`, and the AWS CLI reads `AWS_REGION` -- omitting any one of these causes the pipeline definition to be synthesized with a mismatched role ARN

## Deployment

```bash
# 1. Clone the repository
git clone https://github.com/aws-samples/sample-sagemaker-fleet-cv-pipeline.git
cd sample-sagemaker-fleet-cv-pipeline

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Bootstrap CDK (first time only). The --app override skips app
#    synthesis, which would otherwise fail here: synthesis validates the
#    SageMaker execution role against live IAM, and that role is not
#    created until step 4.
cdk bootstrap --app "echo skip" aws://ACCOUNT_ID/REGION

# 4. First deploy only: create the execution role, bucket, ECR repository,
#    and model package groups WITHOUT the pipeline. The SageMaker SDK
#    validates the execution role against live IAM while building the
#    pipeline definition, so the role must exist before the full deploy.
cdk deploy SageMakerCVPipelineStack -c bootstrap=true --require-approval never

# 5. Deploy both stacks in full (adds the pipeline and orchestration)
cdk deploy --all --require-approval never

# 6. Build and push the custom training container to ECR
./scripts/push_ecr_image.sh
```

## Usage

### Run the pipeline directly

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name anomaly-detection-pipeline-e2e-final
```

### Use real data (BCCD blood cell dataset)

The [BCCD dataset](https://github.com/Shenggan/BCCD_Dataset) (MIT license) contains
364 annotated blood smear microscopy images, originally open-sourced by
[cosmicad](https://github.com/cosmicad/dataset) and
[akshaylamba](https://github.com/akshaylamba/all_CELL_data) and reorganized into
Pascal VOC format by Shenggan. The same dataset is also distributed by
[Roboflow](https://public.roboflow.com/object-detection/bccd) as public domain.
The download script maps platelet presence to the "defect" class as an
illustrative stand-in for a manufacturing defect distribution.

```bash
# Download dataset and upload to S3
python scripts/download_bccd.py --output-dir ./data/bccd --upload-s3

# Run pipeline with real data
aws sagemaker start-pipeline-execution \
  --pipeline-name anomaly-detection-pipeline-e2e-final \
  --pipeline-parameters '[{"Name":"TrainingDataS3Uri","Value":"s3://amzn-s3-demo-bucket/data"}]'
```

### Fleet-scale batch training

Submit training requests for multiple cameras or devices. Step Functions limits concurrent SageMaker pipeline executions using a DynamoDB semaphore.

```bash
# Train 50 camera models (max 5 concurrent)
python scripts/submit_batch.py --cameras 50

# Dry run to preview messages
python scripts/submit_batch.py --cameras 10 --dry-run
```

### Inference from Model Registry

```bash
# Anomaly detection inference
python scripts/inference_demo.py \
  --model-group sm1k-anomaly-model-group --type anomalib

# Defect classifier inference
python scripts/inference_demo.py \
  --model-group sm1k-classifier-model-group --type classifier

# In a fleet run, select a specific camera's model (defaults to "default")
python scripts/inference_demo.py \
  --model-group sm1k-classifier-model-group --type classifier --camera-id cam-001
```

### Edge compilation with SageMaker Neo

Neo compilation is a pipeline step, not a separate process: after each model
passes its quality gate, a `CompileAnomalibNeo` / `CompileClassifierNeo` step
compiles it, and **registration depends on compilation succeeding** (gate →
compile → register), so every Approved package has a device-ready compiled
artifact alongside its ONNX. The compile step repackages the training
artifact ONNX-only (Neo rejects tarballs with extra members), submits the
compilation job under a deterministic name (retries resume the same job
instead of orphaning it), and polls to completion. Compiled artifacts land
under the execution's `neo/` prefix in the pipeline bucket.

Targets are pipeline parameters: `NeoTargetDevice` (e.g. `jetson_xavier`,
`rasp4b`) selects a named device; its default `none` compiles for the
`NeoTargetOs`/`NeoTargetArch` platform (default: LINUX/X86_64, which runs on
any x86_64 Linux host via the bundled
[DLR runtime](https://github.com/neo-ai/neo-ai-dlr)). The exported models use
a dynamic batch axis; Neo requires concrete input shapes, so batch size 1 is
pinned at compilation time (`{"input": [1, 3, 256, 256]}` for the autoencoder,
`[1, 3, 224, 224]` for the classifier). Compilation jobs are free -- you pay
only for the S3 storage of the compiled artifacts.

To compile an already-registered model for additional device targets after
the fact (for example, a new device type joins the fleet), use the ad-hoc
script:

```bash
python scripts/compile_neo.py \
  --model-group sm1k-classifier-model-group --type classifier \
  --target-device jetson_xavier --camera-id cam-001 --region us-east-2
```

## Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TrainingInstanceType` | `ml.m5.xlarge` | Instance type for training jobs |
| `ProcessingInstanceType` | `ml.m5.xlarge` | Instance type for processing jobs |
| `NumSamples` | `400` | Number of synthetic training samples (validation/test are a quarter of this) |
| `MinAUROC` | `0.8` | Anomaly detection quality gate threshold (0.5 = chance) |
| `MinBalancedAccuracy` | `0.6` | Classifier quality gate on balanced accuracy (mean per-class recall); 0.5 = chance |
| `MinDefectRecall` | `0.5` | Classifier quality gate on defect recall -- balanced accuracy alone can pass a model that detects few defects when good-class recall is perfect; both gates must pass |
| `CameraId` | `default` | Fleet device identity: tagged on registered models and used to scope warm-start retrieval |
| `TrainingDataS3Uri` | `none` | S3 URI for real training data (bypasses synthetic generation). Must point at the pipeline's own bucket (as `download_bccd.py --upload-s3` does): the execution role is scoped to that bucket, so external buckets fail with AccessDenied unless you extend the role policy |
| `NeoTargetDevice` | `none` | Named Neo device target (e.g. `jetson_xavier`, `rasp4b`). When set, overrides the OS/arch platform target |
| `NeoTargetOs` | `LINUX` | Neo TargetPlatform OS, used when `NeoTargetDevice` is `none` |
| `NeoTargetArch` | `X86_64` | Neo TargetPlatform architecture, used when `NeoTargetDevice` is `none` |

## Project Structure

```
├── app.py                              # CDK app entry point
├── sagemaker_cv_pipeline_stack.py      # SageMaker Pipeline definition (SDK v3)
├── orchestration_stack.py              # Step Functions + SQS orchestration
├── docker/training/
│   ├── Dockerfile                      # Custom BYOC training container
│   └── entrypoint.py                   # Hyperparameter dispatch entrypoint
├── src/
│   ├── data/preprocess_cv.py           # Data preprocessing (anomalib format)
│   ├── model/
│   │   ├── train_anomalib.py           # Autoencoder anomaly detection training
│   │   ├── train_classifier.py         # ResNet18 defect classifier training
│   │   ├── evaluate_anomalib_onnx.py   # ONNX anomaly model evaluation
│   │   ├── evaluate_classifier_onnx.py # ONNX classifier evaluation
│   │   ├── compile_neo_step.py         # In-pipeline Neo compilation step
│   │   └── fetch_latest_model.py       # Model Registry warm-start fetch
│   └── utils/                          # Shared helpers (tarball, dataset, metrics)
├── scripts/
│   ├── download_bccd.py                # Download BCCD dataset + S3 upload
│   ├── compile_neo.py                  # Ad-hoc Neo compilation (extra targets)
│   ├── submit_batch.py                 # Submit batch training to SQS
│   ├── inference_demo.py               # ONNX inference from Model Registry
│   └── push_ecr_image.sh               # Build + push Docker image to ECR
└── notebooks/
    └── inference_demo.ipynb            # Interactive inference walkthrough
```

## Cost Estimate

Running this sample incurs AWS charges. Key cost drivers:

| Resource | Approximate Cost |
|----------|-----------------|
| SageMaker training (ml.m5.xlarge, two jobs, ~5 min combined at default NumSamples) | ~$0.02 per pipeline run |
| SageMaker processing (ml.m5.xlarge, seven jobs including the two Neo compile drivers at ~4 min each) | ~$0.05 per pipeline run |
| Neo compilation jobs | free (the driving processing jobs above are the cost) |
| S3 storage (training data + model + compiled artifacts) | < $0.10/month |
| Step Functions, Lambda, SQS, DynamoDB, SNS, EventBridge | < $0.10/month (free tier eligible) |

A single pipeline run with synthetic data costs about **$0.07** in compute: 18.9 measured billed instance-minutes of ml.m5.xlarge at ~$0.23/hour on-demand (us-east-2, July 2026), plus small S3/orchestration charges (measured end-to-end runtime ~15 minutes for all 13 steps; steps overlap across the two branches). Fleet-scale runs scale linearly with execution count. Use the [AWS Pricing Calculator](https://calculator.aws/) for estimates in your Region.

## Cleanup

```bash
# 1. Remove model packages from the registry (created at pipeline runtime,
#    so CloudFormation cannot delete them). Include --region if deploying
#    to a non-default region.
REGION="${AWS_REGION:-us-east-1}"

aws sagemaker list-model-packages --model-package-group-name sm1k-anomaly-model-group \
  --region "$REGION" --query 'ModelPackageSummaryList[].ModelPackageArn' --output text | \
  xargs -I {} aws sagemaker delete-model-package --model-package-name {} --region "$REGION"

aws sagemaker list-model-packages --model-package-group-name sm1k-classifier-model-group \
  --region "$REGION" --query 'ModelPackageSummaryList[].ModelPackageArn' --output text | \
  xargs -I {} aws sagemaker delete-model-package --model-package-name {} --region "$REGION"

# 2. Destroy both CDK stacks. Pass -c bootstrap=true so synthesis does not
#    rebuild the pipeline definition (which would fail once the execution
#    role validation runs against a role the destroy is about to remove).
cdk destroy --all -c bootstrap=true --force
```

## Sample Scope Notes

> **Backlog scaling:** the Lambda starts a state-machine execution per queued message immediately, so a large submission becomes many executions polling the semaphore rather than a backlog held in SQS. The slot-wait budget is a stack prop (`max_slot_wait_attempts`, default 480 x 30s = 4 hours); size it as (queued executions / max_concurrency) x average pipeline runtime. Each waiting execution polls on that interval, so state-transition cost grows with backlog size. For sustained fleet-scale backlogs, keep pending work in SQS and admit executions only when capacity exists (a bounded dispatcher) instead of raising the bound.

> **Semaphore limitation:** the concurrency semaphore is a shared DynamoDB counter, and its increment/decrement updates are not idempotent under ambiguous retries -- if an update commits but the response is lost, the Step Functions retry re-applies the arithmetic, which can overcount an acquire (burning a slot until an operator resets `ActiveCount`) or undercount a release. Failures surface through explicit `SemaphoreLeakDetected` / `StopTimeoutDetected` states rather than silently. For production use, replace the counter with per-execution lease records keyed by the state-machine execution ID so acquire and release are idempotent.

> **Model Registry scope:** the pipeline registers models as an artifact catalog -- each package version carries the ONNX artifact, evaluation metrics, and `CameraId` lineage metadata for fleet tracking and warm-start retrieval. Registration only happens after Neo compilation succeeds, so a compiled artifact for the configured target always exists. The registered packages are not directly deployable to SageMaker real-time endpoints or batch transform -- the referenced container is the training image and does not implement a `/ping` / `/invocations` inference server. For edge deployment, use the compiled Neo artifact, or download the ONNX artifact from the package (see `scripts/inference_demo.py` and the notebook) and serve it with ONNX Runtime.

## Security

The SageMaker execution role uses least-privilege inline policies scoped to the S3 bucket, ECR repository, and model package groups this sample deploys, plus the SageMaker training/processing job and CloudWatch Logs permissions those jobs require (job and log ARNs are account- and region-scoped but not name-scoped, since job names are generated at runtime). Review the IAM policies in `sagemaker_cv_pipeline_stack.py` and `orchestration_stack.py` before deploying to production.

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for reporting security issues.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
