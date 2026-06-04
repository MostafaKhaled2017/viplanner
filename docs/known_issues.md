# Known Issues

## Documented In README

- Ubuntu/ROS Noetic may provide `PyYAML 5.3.1` through distutils, while `open3d==0.17.0` requires `PyYAML>=5.4.1`.
- MMCV wheel builds may fail unless installed with the correct CUDA and torch wheel index.
- `pip install -r third_party/mask2former/requirements.txt` may fail with `Invalid version: '1.1-linux32'` when using an older system pip resolver.
- TensorBoard 2.14.x can fail with protobuf 5.x using `MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`.

## Observed In This Checkout

- `rg` reports a parse error for `.gitignore` because line 9 contains a dangling escaped pattern: `AGENTS.md\`.
- The repository has generated or build output directories checked out locally: `build/`, `devel/`, and `logs/`.
- Some config files contain absolute paths under `/workspaces/viplanner`, so portability depends on either editing configs or matching that workspace path.

## Warm-start sweep runs can stop before improving their own validation loss

**Symptom**

Hyperparameter sweep trials launched with `resume: true` and a shared external `resume_model_path` can stop after `early_stop_patience` epochs while visible per-batch training loss is still decreasing. Logs show every epoch incrementing the early-stopping counter and no `Save model of epoch ...` line.

**Context**

This was observed with:

```bash
python3 viplanner/tune_train.py --sweep-config viplanner/config/sweep.yaml --gpus 1,2,3 --max-parallel 6
```

**Likely cause**

The trainer loaded both weights and the source checkpoint's stored best validation loss. Sweep trials with changed data or loss settings then had to beat the source checkpoint's validation loss before early stopping could reset or save a model.

**Fix or workaround**

External resume checkpoints are now treated as warm starts: weights are loaded, the checkpoint is copied into the run directory, and `best_loss` is reset for the new run. True resumes from the run's own `model.pt` still preserve the stored best validation loss.

**Validation**

`pytest tests/test_training_early_stopping.py` passed. `PYTHONPYCACHEPREFIX=/tmp/viplanner-pycache python3 -m py_compile viplanner/utils/trainer.py tests/test_training_early_stopping.py` passed.

**Related files**

- `viplanner/utils/trainer.py`
- `tests/test_training_early_stopping.py`

## Parallel sweep can delete generated warped semantic images

**Symptom**

Training can fail in a DataLoader worker with `FileNotFoundError` for a generated image such as `src/planner/data/forest_s3/img_warp/0177_cam1.png`.

**Context**

This can happen during parallel hyperparameter sweeps where multiple training processes share the same dataset environment directories.

**Likely cause**

`PlannerDataGenerator` previously wrote warped semantic inputs into a shared per-environment `img_warp` directory, and cleanup removed that whole directory. A completed trial could delete files still referenced by another active trial.

**Fix or workaround**

Generated warped semantic images and generated depth-edge images are now written under per-generator subdirectories, and cleanup removes only the generator-owned subdirectory. As a workaround on older code, reduce sweep parallelism or avoid cleanup while parallel trials are active.

**Validation**

`pytest tests/test_dataset_generated_dirs.py` passed.

**Related files**

- `viplanner/utils/dataset.py`
- `tests/test_dataset_generated_dirs.py`

## Unknown

- Whether all ROS launch files run in the current container without external simulator, robot, camera, TF, and model assets.
- Whether all datasets referenced in configs are present in this checkout.
