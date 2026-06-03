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

## Unknown

- Whether all ROS launch files run in the current container without external simulator, robot, camera, TF, and model assets.
- Whether all datasets referenced in configs are present in this checkout.
