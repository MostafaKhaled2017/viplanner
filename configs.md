# Learning Config Reference

This file explains the editable YAML parameters used by [`train.yaml`](/workspaces/viplanner/viplanner/config/train.yaml) and defined by [`learning_cfg.py`](/workspaces/viplanner/viplanner/config/learning_cfg.py). It also summarizes the related cost-map and semantic metadata definitions in [`costmap_cfg.py`](/workspaces/viplanner/viplanner/config/costmap_cfg.py), [`viplanner_sem_meta.py`](/workspaces/viplanner/viplanner/config/viplanner_sem_meta.py), and [`coco_sem_meta.py`](/workspaces/viplanner/viplanner/config/coco_sem_meta.py).

The training YAML contains two config layers:

- `config`: top-level training settings from `TrainCfg`
- `config.data_cfg`: dataset construction and augmentation settings from `DataCfg`

Two names in `learning_cfg.py`, `depth_suffix` and `sem_suffix`, are fixed class attributes rather than dataclass constructor fields, so they are not included in `train.yaml`.

## TrainCfg

### High-level run settings

- `sem`: Use semantic images as the second visual input branch alongside depth.
- `rgb`: Use RGB images as the second visual input branch alongside depth. `sem` and `rgb` should not both be `true`.
- `file_name`: Optional suffix added to the saved model name. If `null`, the YAML filename is used when loading through `from_yaml()`.
- `seed`: Random seed for reproducibility-sensitive parts of training and testing.
- `gpu_id`: CUDA device index used for training and evaluation.
- `file_path`: Base experiment directory containing `models/`, `data/`, and `logs/`. The environment variable `EXPERIMENT_DIRECTORY` overrides it when set.

### Data and environment selection

- `cost_map_name`: Name of the cost map directory/file to load for each environment, for example `cost_map_sem` or `cost_map_geom`.
- `env_list`: Environment IDs to use. Each entry is expected under `data/<env_name>`.
- `test_env_id`: Index inside `env_list` reserved for validation/test. Training skips this environment and testing uses only this environment.
- `data_cfg`: Nested dataset configuration. This can be a single `DataCfg` shared across all environments.
- `multi_epoch_dataloader`: Declared config flag for multi-epoch loading. It is currently present in the schema but not actively used in the trainer path.
- `num_workers`: Number of PyTorch dataloader worker processes.
- `load_in_ram`: Preload all selected samples into memory before iteration to reduce repeated disk reads.

### Loss settings

- `fear_ahead_dist`: Lookahead distance used by the fear-path logic during trajectory evaluation.
- `w_obs`: Weight of obstacle cost in the trajectory loss.
- `w_height`: Weight of height-related cost in the trajectory loss.
- `w_motion`: Weight of motion smoothness cost in the trajectory loss.
- `w_goal`: Weight of goal-reaching cost in the trajectory loss.
- `obstacle_thread`: Obstacle threshold used to decide whether a path is considered unsafe. The field name is spelled `thread` in code, but it acts as a threshold.

### Network settings

- `img_input_size`: Input image size after preprocessing, in `[height, width]`.
- `in_channel`: Number of channels used to encode the goal input in the decoder.
- `knodes`: Number of predicted waypoint nodes in the output trajectory.
- `pre_train_sem`: Enable the pretrained visual encoder path used when `rgb: true`.
- `pre_train_cfg`: Relative path under `models/` to the pretrained Mask2Former-style config file.
- `pre_train_weights`: Relative path under `models/` to pretrained encoder weights.
- `pre_train_freeze`: Freeze the pretrained RGB encoder instead of fine-tuning it.
- `decoder_small`: Use the smaller decoder variant with fewer parameters.

### Training schedule

- `resume`: Resume training from an existing checkpoint in the current model directory.
- `epochs`: Maximum number of training epochs.
- `batch_size`: Batch size for both training and validation loaders.
- `hierarchical`: Enable hierarchical sampling, which gradually shifts sample selection away from pure FOV examples.
- `hierarchical_step`: Number of epochs between hierarchical sampling updates.
- `hierarchical_front_step_ratio`: Amount added to the front-sample ratio at each hierarchical step.
- `hierarchical_back_step_ratio`: Amount added to the back-sample ratio at each hierarchical step.

### Optimizer and scheduler

- `lr`: Initial learning rate.
- `factor`: Multiplicative decay used by the scheduler when validation loss plateaus.
- `min_lr`: Lower bound for scheduler-driven learning-rate decay.
- `patience`: Number of plateau epochs before reducing the learning rate.
- `optimizer`: Optimizer type. Supported values are `sgd` and `adam`.
- `momentum`: SGD momentum value. Used only when `optimizer: sgd`.
- `w_decay`: Weight decay coefficient.

### Visualization and logging

- `camera_tilt`: Camera tilt angle used by visualization utilities.
- `n_visualize`: Number of trajectories to visualize during evaluation/debug output.
- `wb_project`: Weights & Biases project name.
- `wb_entity`: Weights & Biases entity or team name.
- `wb_api_key`: Weights & Biases API key used by the trainer process.

## DataCfg

### Dataset source flags

- `real_world_data`: Rotate loaded images by 180 degrees to match the expected real-world camera orientation.
- `carla`: Apply CARLA-specific filtering behavior for large open-space regions.

### Depth and goal filtering

- `max_depth`: Maximum usable depth value in meters. Larger values are zeroed out.
- `max_goal_distance`: Maximum allowed odom-to-goal distance used for sample generation.
- `min_goal_distance`: Minimum allowed odom-to-goal distance used for sample generation.
- `distance_scheme`: Distribution of samples over distance buckets. Keys are distances and values are percentages of the sample budget. Distances should be ordered and aligned with the intended maximum range.
- `obs_cost_height`: Starting poses above this cost threshold are filtered out as unsuitable.
- `fov_scale`: Scale factor applied to the field-of-view constraint when selecting valid goals.
- `depth_scale`: Depth-image scaling factor used by the dataset pipeline.

### Train/validation split and sample composition

- `ratio`: Fraction of generated pairs assigned to training; the remainder goes to validation.
- `max_train_pairs`: Optional hard cap on the number of generated training pairs.
- `pairs_per_image`: Target number of odom-goal pairs to generate per recorded image.
- `ratio_fov_samples`: Fraction of samples whose goals stay inside the robot field of view.
- `ratio_front_samples`: Fraction of samples whose goals are in front of the robot but outside the field of view.
- `ratio_back_samples`: Fraction of samples whose goals are behind the robot.

The three ratio fields above should sum to `1.0`.

### Edge and sensor-noise augmentation

- `noise_edges`: Apply edge blur augmentation intended to mimic depth artifacts near object boundaries.
- `edge_threshold`: Threshold used by the edge-noise augmentation logic.
- `extend_kernel_size`: Kernel size used when expanding noisy edge regions.

### Depth-image augmentation

- `depth_salt_pepper`: Proportion of depth pixels replaced with salt-and-pepper noise.
- `depth_gaussian`: Variance-like parameter used for Gaussian depth noise in the current implementation.
- `depth_random_polygons_nb`: Number of random blackout polygons to paint onto depth images.
- `depth_random_polygon_size`: Maximum size used when generating random depth polygons.

### Semantic/RGB image augmentation

- `sem_rgb_pepper`: Proportion of semantic or RGB pixels replaced with pepper noise.
- `sem_rgb_black_img`: Probability of turning an entire semantic or RGB image black.
- `sem_rgb_random_polygons_nb`: Number of random blackout polygons added to semantic or RGB images.
- `sem_rgb_random_polygon_size`: Maximum size used when generating semantic or RGB polygons.

## Cost Map Configs

[`costmap_cfg.py`](/workspaces/viplanner/viplanner/config/costmap_cfg.py) defines the dataclasses used when reconstructing point clouds and building semantic or geometric cost maps.

Like `TrainCfg`, these values are intended to be YAML-serializable. Reconstruction settings are loaded from the top-level `reconstruction:` section of the shared [`costmap.yaml`](/workspaces/viplanner/viplanner/config/costmap.yaml) file when using [`depth_reconstruct.py`](/workspaces/viplanner/viplanner/depth_reconstruct.py).

### ReconstructionCfg

Used for depth or semantic point-cloud reconstruction before cost-map generation.

- `data_dir`: Root directory containing environment folders.
- `env`: Environment name under `data_dir`.
- `depth_suffix`: Optional suffix appended to depth filenames and depth extrinsic files.
- `sem_suffix`: Optional suffix appended to semantic filenames and semantic extrinsic files.
- `high_res_depth`: Use higher-resolution depth images for reconstruction when available.
- `voxel_size`: Voxel size in meters for reconstruction/downsampling.
- `start_idx`: First image index to include.
- `max_images`: Maximum number of images to reconstruct. Use `null` to process all images.
- `depth_scale`: Scale factor applied to depth values during reconstruction.
- `semantics`: Include semantic information during reconstruction.
- `point_cloud_batch_size`: Number of images processed into 3D points per batch. Higher values trade memory for speed.

The helper methods `get_data_path()` and `get_out_path()` build environment-specific input and output paths from the configured directories.

### GeneralCostMapConfig

Shared map-generation settings used by both semantic and TSDF cost maps.

- `root_path`: Environment directory containing the point cloud input.
- `ply_file`: Point-cloud filename within `root_path`.
- `resolution`: Output cost-map resolution in meters per cell.
- `clear_dist`: Extra border around the observed point-cloud extent to keep planned paths inside the map.
- `sigma_smooth`: Gaussian smoothing strength applied to the generated cost map.
- `x_min`, `y_min`, `x_max`, `y_max`: Optional manual map bounds. If `null`, bounds are taken from the point cloud.

### SemCostMapConfig

Settings specific to semantic cost-map generation from labeled point clouds.

- `ground_height`: Minimum allowed ground height. Use `null` to disable that filter.
- `robot_height`: Robot body height in meters.
- `robot_height_factor`: Multiplier used when filtering points above the robot.
- `nb_neighbors`: Neighbor count used for point-cloud outlier filtering.
- `std_ratio`: Standard-deviation threshold for point-cloud outlier removal.
- `downsample`: Enable point-cloud downsampling before map generation.
- `nb_neigh`: Neighbor count used during smoothing iterations.
- `change_decimal`: Decimal precision used when measuring convergence.
- `conv_crit`: Fraction of points that must still change to keep iterating.
- `nb_tasks`: Parallel task count for processing. Use `null` to use all available cores.
- `sigma_smooth`: Smoothing strength used inside the semantic map pipeline.
- `max_iterations`: Maximum number of smoothing/refinement iterations.
- `obstacle_threshold`: Threshold, expressed relative to the maximum semantic class loss, for marking obstacles.
- `negative_reward`: Offset applied to the lowest-cost traversable space to introduce a gradient toward the center of free space.
- `round_decimal_traversable`: Decimal precision used when selecting cells treated as fully traversable.
- `compute_height_map`: Also generate a height map alongside the semantic cost map.

### TsdfCostMapConfig

Settings specific to geometry-only cost maps built from TSDF-style point clouds.

- `offset_z`: Vertical offset applied to the point cloud before filtering.
- `ground_height`: Ground-removal threshold in meters.
- `robot_height`: Robot body height in meters.
- `robot_height_factor`: Multiplier used when filtering points above the robot.
- `nb_neighbors`: Neighbor count used for geometric outlier filtering.
- `std_ratio`: Standard-deviation threshold used during outlier filtering.
- `filter_outliers`: Enable statistical outlier removal.
- `sigma_expand`: Expansion/dilation strength for obstacle regions.
- `obstacle_threshold`: Value above which cells are treated as obstacles.
- `free_space_threshold`: Value below which cells are treated as free space.

### CostMapConfig

Top-level wrapper that selects which cost-map pipeline to run and bundles the nested configs above.

- `semantics`: Enable semantic cost-map generation.
- `geometry`: Enable geometry/TSDF cost-map generation.
- `map_name`: Output name for the generated cost map.
- `general`: Nested [`GeneralCostMapConfig`](#generalcostmapconfig).
- `sem_cost_map`: Nested [`SemCostMapConfig`](#semcostmapconfig).
- `tsdf_cost_map`: Nested [`TsdfCostMapConfig`](#tsdfcostmapconfig).
- `visualize`: Render the generated cost map for inspection.
- `x_start`, `y_start`: Filled in by the code at runtime. They are marked "do not change" in the config module.

## VIPlanner Semantic Metadata

[`viplanner_sem_meta.py`](/workspaces/viplanner/viplanner/config/viplanner_sem_meta.py) defines the semantic classes and traversal costs used by the semantic cost-map pipeline.

### Loss constants

- `OBSTACLE_LOSS = 2.0`: Assigned to obstacle-like classes.
- `TRAVERSABLE_INTENDED_LOSS = 0`: Assigned to preferred traversable classes such as sidewalks and floors.
- `TRAVERSABLE_UNINTENDED_LOSS = 0.5`: Assigned to traversable but less preferred terrain such as gravel or snow.
- `ROAD_LOSS = 1.5`: Assigned to roads.
- `TERRAIN_LOSS = 1.0`: Assigned to terrain-like surfaces and some soft indoor surfaces.

### VIPLANNER_SEM_META

`VIPLANNER_SEM_META` is the core class table. Each entry contains:

- `name`: Canonical VIPlanner semantic class name.
- `loss`: Traversal cost assigned to that class.
- `color`: RGB visualization color.
- `ground`: Whether the class is considered ground-like during semantic map processing.

The defined classes are:

- Intended traversable: `sidewalk`, `crosswalk`, `floor`, `stairs`
- Traversable but less preferred: `gravel`, `sand`, `snow`
- Terrain-like: `indoor_soft`, `terrain`, `road`
- Dynamic or object obstacles: `person`, `anymal`, `vehicle`, `on_rails`, `motorcycle`, `bicycle`
- Structural obstacles: `building`, `wall`, `fence`, `bridge`, `tunnel`
- Small or scene obstacles: `pole`, `traffic_sign`, `traffic_light`, `bench`, `vegetation`, `water_surface`, `sky`, `background`, `dynamic`, `static`, `furniture`, `door`, `ceiling`

### VIPlannerSemMetaHandler

`VIPlannerSemMetaHandler` builds convenient lookup tables from `VIPLANNER_SEM_META`:

- `class_loss`: Map from class name to traversal loss.
- `class_color`: Map from class name to RGB color.
- `class_ground`: Map from class name to ground flag.
- `class_id`: Map from class name to its index in `VIPLANNER_SEM_META`.

It also provides:

- `get_colors_for_names(name_list)`: Returns RGB colors for the provided class names.
- `colors`: Ordered list of class colors.
- `losses`: Ordered list of class losses.
- `names`: Ordered list of class names.
- `ground`: Ordered list of ground flags.

## COCO Semantic Mapping

[`coco_sem_meta.py`](/workspaces/viplanner/viplanner/config/coco_sem_meta.py) bridges raw COCO semantic classes to the smaller VIPlanner semantic vocabulary.

### COCO_CATEGORIES

`COCO_CATEGORIES` is the full COCO class table used as input metadata. Each entry stores:

- `id`: COCO category ID.
- `name`: COCO category name.
- `color`: RGB visualization color.
- `isthing`: COCO "thing" vs. "stuff" flag.

### _COCO_MAPPING

`_COCO_MAPPING` groups one or more COCO class names under each VIPlanner semantic class. Examples:

- `road` maps from COCO `road`
- `sidewalk` maps from `pavement-merged`
- `floor` maps from `floor-other-merged`, `floor-wood`, `platform`, `playingfield`, and `rug-merged`
- `vehicle` maps from `car`, `bus`, `truck`, and `boat`
- `vegetation` maps from `potted plant`, `flower`, `tree-merged`, `mountain-merged`, and `rock-merged`
- `dynamic` groups many movable, handheld, kitchen, food, electronics, and miscellaneous object classes
- `static` covers otherwise static background classes such as `banner`, `cardboard`, `light`, `tent`, and `unknown`

The mapping is intentionally many-to-one: multiple COCO labels collapse into a single VIPlanner cost-map class.

### Helper functions

- `get_class_for_id()`: Builds a dictionary from the index in `COCO_CATEGORIES` to the corresponding VIPlanner class name. Matching is done by checking whether any configured mapping keyword appears in the COCO category name.
- `get_class_for_id_mmdet(class_list)`: Same idea, but uses a provided MMDetection-style class-name list instead of `COCO_CATEGORIES`.

Both helpers print a message when no VIPlanner mapping is found for a class.
