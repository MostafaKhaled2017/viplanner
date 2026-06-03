# Parameters

## Planner Package (`src/planner/config/anymal_mount.yaml`)

- `main_freq`
- `image_flip`
- `conv_dist`
- `max_depth`
- `overlap_ratio_thres`
- `depth_zero_ratio_thres`
- `model_save`
- `m2f_model_path`
- `m2f_cfg_file`
- `depth_topic`
- `depth_info_topic`
- `rgb_topic`
- `rgb_info_topic`
- `mount_cam_frame`
- `goal_topic`
- `path_topic`
- `m2f_timer_topic`
- `depth_uint_type`
- `compressed`
- `robot_id`
- `world_id`
- `is_fear_act`
- `buffer_size`
- `angular_thread`
- `track_dist`
- `joyGoal_scale`

## Standalone ROS 1 Planner (`src/viplanner_ros1/config/viplanner.yaml`)

- `main_freq`
- `verbose`
- `use_sim_time`
- `model_save`
- `depth_topic`
- `depth_width`
- `depth_height`
- `rgb_topic`
- `rgb_width`
- `rgb_height`
- `rgb_compressed`
- `goal_topic`
- `path_topic`
- `fear_topic`
- `robot_id`
- `world_id`
- `mount_cam_frame`
- `depth_uint_type`
- `max_depth`
- `image_flip`
- `conv_dist`
- `m2f_config_path`
- `m2f_model_path`
- `m2f_device`
- `is_fear_act`
- `fear_threshold`
- `buffer_size`
- `angular_thread`
- `track_dist`
- `joyGoal_scale`
- `subgoal_max_distance`
- `subgoal_update_distance`
- `debug_enabled`
- `debug_dump_dir`
- `debug_dump_every_n`
- `debug_save_input_tensors`
- `debug_height_warn_threshold`
- `use_camera_frame_goal`

## Data Collection (`src/planner/config/data_collect_sim.yaml`)

- `dataset_root`
- `env_name`
- `depth_topic`
- `rgb_topic`
- `depth_info_topic`
- `rgb_info_topic`
- `rgb_compressed`
- `world_frame_id`
- `depth_frame_id`
- `rgb_frame_id`
- `rgb_pose_fallback_to_depth`
- `use_latest_tf`
- `tf_buffer_seconds`
- `tf_timeout_seconds`
- `convert_optical_to_viplanner`
- `depth_suffix`
- `rgb_suffix`
- `depth_width`
- `depth_height`
- `rgb_width`
- `rgb_height`
- `depth_is_uint16`
- `depth_scale`
- `max_depth`
- `save_depth_npy`
- `sync_slop_seconds`
- `sample_min_interval`
- `sample_min_distance`
- `sample_min_angle_deg`
- `max_samples`

## Planner Validation (`src/planner_validation_ros1/config/planner_validation.yaml`)

- `main_freq`
- `validation_paths_file`
- `log_dir`
- `planner_id`
- `run_id`
- `goal_topic`
- `start_pose_topic`
- `planner_status_topic`
- `collision_topic`
- `robot_id`
- `world_id`
- `start_yaw_offset_rad`
- `trial_timeout_sec`
- `start_arrival_thresh`
- `goal_arrival_thresh`
- `debug_csv_columns`
- `shutdown_on_complete`

## Path Follower

Configured by launch arguments and private parameters in `src/pathFollower/launch/path_follower.launch`, including odometry topic, command topic, base frame, tracking distance, speed, acceleration, stop thresholds, joystick delay, and drive mode.

## Visualization

Configured by `src/visualizer/config/*.yaml`, including visualization output topic, image topic, camera-info topic, goal topic, path topic, robot frame, odom frame, domain, image flip, and max depth.
