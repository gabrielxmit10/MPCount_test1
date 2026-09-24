# MPCount + MovingDroneCrowd++ Colab guide

For the first session, run `colab_mdc_mpcount.ipynb` only through section 8, **One-batch training smoke test**. That verifies the complete MPCount/MDC pipeline without starting a long experiment.

## Before opening Colab

Push the modified MPCount repository to:

```text
https://github.com/gabrielxmit10/MPCount_test1.git
```

The notebook clones branch `main`.

The MDC archive is reused from the P2PNet Drive project. It must already exist at:

```text
MyDrive/P2PNet_MDC/data/MovingDroneCrowd++.tar
```

MPCount does not write anything else inside `P2PNet_MDC`.

Upload the existing MPCount ShanghaiTech-A weight:

```text
local:  logs/sta/sta_deterministic.pth
Drive:  MyDrive/MPCount/checkpoints/sta_deterministic.pth
```

The notebook creates `MyDrive/MPCount/runs/` automatically.

## First-session configuration

Open `colab_mdc_mpcount.ipynb` in VS Code. In the central configuration cell, verify:

```python
REPO_URL = "https://github.com/gabrielxmit10/MPCount_test1.git"
REPO_REF = "main"
REPO_DIR = Path("/content/MPCount_MDC")

DRIVE_PROJECT = Path("/content/drive/MyDrive/MPCount")
DATA_SOURCE = "archive"
DATASET_ARCHIVE = Path(
    "/content/drive/MyDrive/P2PNet_MDC/data/MovingDroneCrowd++.tar"
)
DATASET_ROOT = Path("/content/data/MovingDroneCrowd++")

SHANGHAITECH_WEIGHTS = (
    DRIVE_PROJECT / "checkpoints/sta_deterministic.pth"
)

TRAIN_INITIALIZATION = "shanghaitech"
RUN_NAME = "mpcount_mdc_run_001"
```

Keep every expensive-operation switch disabled:

```python
RUN_FULL_TRAINING = False
RUN_RESUME_TRAINING = False
RUN_FINAL_EVALUATION = False
RUN_BATCH_INFERENCE = False
```

Using ShanghaiTech in sections 5-7 is only a functionality check. The initialization for a real experiment remains a separate decision.

## Sections 1-8

### 1. Mount Drive and inspect the runtime

Connect the notebook to a Google Colab GPU runtime before running this cell. A T4 is sufficient for the smoke tests.

The cell:

- mounts Google Drive;
- creates the MPCount run directory;
- prints Python and `nvidia-smi` information;
- reports free `/content` storage.

Verify that a GPU is present. Dataset staging needs approximately 26 GB of free runtime storage while the tar file and extracted data temporarily coexist.

### 2. Clone the modified repository

This cell:

- clones `MPCount_test1` to `/content/MPCount_MDC`;
- fetches and fast-forwards `main` when a runtime checkout already exists;
- installs `requirements_colab.txt` without replacing Colab's Torch/Torchvision pair;
- prints the exact Git commit, Torch versions, and CUDA status;
- defines the initialization helpers and command logger;
- writes `run_manifest.json`.

Verify:

```text
CUDA: True
```

If the GitHub repository is private, the anonymous clone will fail. Make it public or add an authenticated clone method before continuing.

### 3. Stage MovingDroneCrowd++

This cell reads only the dataset archive from the P2PNet Drive project:

```text
/content/drive/MyDrive/P2PNet_MDC/data/MovingDroneCrowd++.tar
```

It copies the tar to `/content`, extracts it, deletes only the temporary runtime tar, and leaves the Drive archive untouched.

Expected result:

```text
Dataset ready: /content/data/MovingDroneCrowd++
```

Re-running the cell in the same runtime reuses the extracted dataset rather than copying it again.

### 4. Validate all splits and inspect one sample

Run both cells in this section.

Expected validation totals:

| Split | Clips | Frames | Head annotations |
|---|---:|---:|---:|
| train.txt | 64 | 4,011 | 351,862 |
| val.txt | 15 | 724 | 64,888 |
| test.txt | 41 | 2,462 | 221,968 |

The validator should report zero structural errors. Boundary warnings are expected and handled by the adapter.

The report is written to:

```text
MyDrive/MPCount/runs/<RUN_NAME>/dataset_validation.json
```

The next cell displays the first validation frame with the derived MPCount head-center points. Check that the points are visually sensible.

### 5. Cheap model/data smoke test

This uses:

- a real MDC training sample;
- a 128x128 crop;
- the ShanghaiTech-A `sta_deterministic.pth` checkpoint;
- strict checkpoint loading;
- a complete model forward pass.

Expected messages include:

```text
Dataset OK
Loader OK
Checkpoint OK
Model forward OK
```

### 6. Single-image inference

This runs the ShanghaiTech checkpoint on:

```text
/content/data/MovingDroneCrowd++/frames/scene_4/1/1.jpg
```

It displays the generated visualization and saves:

```text
MyDrive/MPCount/runs/<RUN_NAME>/smoke_inference/
├── counts.txt
├── 1.png
└── 1_pred_dmap.npy
```

The numerical count is not an MDC accuracy result because the weight has not been adapted to MDC++.

### 7. Tiny two-frame MDC evaluation

This evaluates two validation frames using the ShanghaiTech checkpoint. It verifies MDC ground truth, tiled prediction, and metric writing.

Outputs:

```text
MyDrive/MPCount/runs/<RUN_NAME>/smoke_evaluation/
├── metrics.json
└── predictions.csv
```

The two-frame MAE/RMSE is only a pipeline check.

### 8. One-batch training smoke test

This is the first cell controlled by `TRAIN_INITIALIZATION`.

With the default:

```python
TRAIN_INITIALIZATION = "shanghaitech"
```

it verifies actual fine-tuning from `sta_deterministic.pth`.

The cell performs:

- a two-sample 128x128 MDC batch;
- one loss/backpropagation pass;
- one AdamW optimizer update;
- checkpoint serialization.

The disposable checkpoint is:

```text
/content/mpcount_training_smoke/checkpoints/latest.pth
```

Success is confirmed by:

```text
Tiny training step OK
Tiny training checkpoint created successfully
```

Stop after section 8 for the first session. Running later guarded cells while their switches are false is safe; they print that they were skipped.

## Choosing the real training initialization

Three choices are implemented.

### ShanghaiTech-A fine-tuning

```python
TRAIN_INITIALIZATION = "shanghaitech"
```

The complete MPCount checkpoint is loaded, including the VGG encoder and MPCount-specific memory/density/classification parameters. This is the path directly verified by the default section-8 smoke test.

### ImageNet initialization

```python
TRAIN_INITIALIZATION = "imagenet"
```

Torchvision loads ImageNet VGG16-BN weights. MPCount-specific heads and memory start from their normal random initialization.

Before full ImageNet-initialized training:

1. change `TRAIN_INITIALIZATION` to `"imagenet"`;
2. rerun the central configuration cell;
3. rerun section 2 so the helper sees the new setting;
4. rerun section 8;
5. verify that the one-batch checkpoint is created.

### Fully random initialization

```python
TRAIN_INITIALIZATION = "random"
```

No ShanghaiTech or ImageNet weight is loaded. This is available for controlled comparisons but is not the recommended first experiment.

## Full training

Only after the selected initialization passes section 8:

```python
RUN_NAME = "mpcount_mdc_shanghaitech_001"  # or an imagenet-specific name
TRAIN_INITIALIZATION = "shanghaitech"      # or "imagenet"
RUN_FULL_TRAINING = True
```

Keep the other switches false, rerun the configuration cell, rerun section 2 so `RUN_DIR`/helpers/manifest are refreshed, confirm the dataset remains staged, and run section 9.

Current actual settings are:

```python
EPOCHS = 180
LEARNING_RATE = 0.001
BATCH_SIZE = 2
CROP_SIZE = 320
PATCH_SIZE = 512
NUM_WORKERS = 2
```

MPCount runs the complete 724-frame validation set after every epoch. Best-model selection uses validation MAE.

Persistent training outputs:

```text
MyDrive/MPCount/runs/<RUN_NAME>/
├── best.pth
├── last.pth
├── last_resume.pth
├── log.txt
├── mdc_train.yml
├── resolved_config.yml
├── commands.log
├── dataset_validation.json
└── run_manifest.json
```

`last_resume.pth`, the log, resolved config, and any improved best model are synchronized after each completed epoch. `last.pth` is copied after a clean training finish.

## Resume after reset

Use the same `RUN_NAME` and keep `EPOCHS` equal to the original final target:

```python
RUN_FULL_TRAINING = False
RUN_RESUME_TRAINING = True
```

Rerun sections 1-3 in the fresh runtime, then section 10. It loads:

```text
MyDrive/MPCount/runs/<RUN_NAME>/last_resume.pth
```

Do not start a separate new run when resuming.

## Final evaluation

Start with validation:

```python
FINAL_SPLIT = VAL_SPLIT
RUN_FINAL_EVALUATION = True
```

After all choices are frozen, use the held-out test set once:

```python
FINAL_SPLIT = TEST_SPLIT
```

Evaluation creates a separate result directory such as:

```text
MyDrive/MPCount/runs/<RUN_NAME>_validation_full/
MyDrive/MPCount/runs/<RUN_NAME>_test_full/
```

Each contains `log.txt` with MAE/RMSE and `test_predictions.csv` with every frame's prediction, ground truth, and absolute error.

## Batch inference

Choose `INFERENCE_INPUT`, then set:

```python
RUN_BATCH_INFERENCE = True
```

The default input is the MDC clip:

```text
/content/data/MovingDroneCrowd++/frames/scene_1/1
```

Outputs are copied to:

```text
MyDrive/MPCount/runs/<RUN_NAME>/batch_inference/
```

They include `counts.txt`, one visualization PNG per image, and one raw predicted-density `.npy` per image.

## TensorBoard

The current MPCount trainer does not emit TensorBoard events. Use `log.txt`, YAML configs, `commands.log`, `run_manifest.json`, CSV/JSON metrics, checkpoints, and saved inference artifacts.
