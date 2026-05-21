import ast
import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[1] / "src" / "planner_validation_ros1"
sys.path.insert(0, str(PKG_ROOT / "src"))

from planner_validation_ros1.validation_core import (  # noqa: E402
    append_result_row,
    append_sweep_summary_row,
    append_sweep_trial_rows,
    build_sweep_configs,
    compute_sweep_summary,
    discover_model_directories,
    distance3,
    init_results_csv,
    load_scenarios_txt,
    read_csv_dicts,
    yaw_from_start_to_goal,
    yaw_to_quaternion_zw,
)


class PlannerValidationRos1PackageTest(unittest.TestCase):
    def test_scenario_parser_accepts_comments_and_rejects_bad_column_counts(self):
        warnings = []
        text = """
# pair_id start_x start_y start_z goal_x goal_y goal_z
1 0.0 1.0 2.0 3.0 4.0 5.0

bad 1 2
2 -1.0 -2.0 -3.0 4.0 5.0 6.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scenarios.txt"
            path.write_text(text)
            scenarios = load_scenarios_txt(str(path), warn_fn=warnings.append)

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0].pair_id, 1)
        self.assertEqual(scenarios[0].start, (0.0, 1.0, 2.0))
        self.assertEqual(scenarios[0].goal, (3.0, 4.0, 5.0))
        self.assertEqual(scenarios[1].pair_id, 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("expected 7 columns", warnings[0])

    def test_distance_and_yaw_helpers(self):
        self.assertAlmostEqual(distance3((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)), 13.0)
        self.assertAlmostEqual(yaw_from_start_to_goal((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), math.pi / 2.0)
        qz, qw = yaw_to_quaternion_zw(math.pi)
        self.assertAlmostEqual(qz, 1.0)
        self.assertAlmostEqual(qw, 0.0, places=7)

    def test_ros1_yaml_is_flat(self):
        config_path = PKG_ROOT / "config" / "planner_validation.yaml"
        data = yaml.safe_load(config_path.read_text())
        self.assertIn("main_freq", data)
        self.assertIn("validation_paths_file", data)
        self.assertFalse(data["debug_csv_columns"])
        self.assertFalse(data["shutdown_on_complete"])
        self.assertNotIn("/**", data)
        self.assertNotIn("ros__parameters", config_path.read_text())

    def test_debug_result_columns_are_optional(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = init_results_csv(tmp_dir, include_debug_columns=True)
            append_result_row(
                csv_path,
                pair_id=1,
                planner_id="viplanner",
                collided=False,
                reached_goal=False,
                distance_to_goal=2.5,
                elapsed_sec=30.0,
                debug_values={
                    "start_z": 1.0,
                    "goal_z": 1.0,
                    "robot_start_z": 1.0,
                    "robot_end_z": 0.9,
                    "final_distance": 2.5,
                    "planner_status": 0,
                    "outcome": "timeout",
                },
            )
            lines = Path(csv_path).read_text().splitlines()

        self.assertIn("robot_start_z", lines[0])
        self.assertIn("planner_status", lines[0])
        self.assertIn("timeout", lines[1])

    def test_model_discovery_uses_immediate_valid_children_in_sorted_order(self):
        warnings = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            valid_b = root / "b_model"
            valid_a = root / "a_model"
            invalid = root / "c_missing_config"
            nested = root / "nested" / "inner_model"
            for path in (valid_b, valid_a, invalid, nested):
                path.mkdir(parents=True)
                (path / "model.pt").touch()
            (valid_b / "model.yaml").touch()
            (valid_a / "model.yaml").touch()
            (nested / "model.yaml").touch()

            models = discover_model_directories(str(root), warn_fn=warnings.append)

        self.assertEqual([path.name for path in models], ["a_model", "b_model"])
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("c_missing_config" in warning for warning in warnings))
        self.assertTrue(any("nested" in warning for warning in warnings))

    def test_sweep_config_overrides_runtime_values(self):
        viplanner_config = {"model_save": "/old/model", "depth_topic": "/depth"}
        validation_config = {"planner_id": "base", "planner_status_topic": "/old_status", "trial_timeout_sec": 10.0}

        viplanner_run_config, validation_run_config = build_sweep_configs(
            viplanner_config,
            validation_config,
            model_path="/models/model_a",
            model_name="model_a",
            model_log_dir="/tmp/sweep/model_a",
        )

        self.assertEqual(viplanner_run_config["model_save"], "/models/model_a")
        self.assertEqual(viplanner_run_config["depth_topic"], "/depth")
        self.assertEqual(validation_run_config["log_dir"], "/tmp/sweep/model_a")
        self.assertEqual(validation_run_config["planner_id"], "model_a")
        self.assertEqual(validation_run_config["planner_status_topic"], "/viplanner/status")
        self.assertTrue(validation_run_config["shutdown_on_complete"])
        self.assertEqual(validation_run_config["trial_timeout_sec"], 10.0)

    def test_sweep_trial_merge_adds_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_csv = init_results_csv(tmp_dir)
            append_result_row(
                validation_csv,
                pair_id=7,
                planner_id="model_a",
                collided=True,
                reached_goal=True,
                distance_to_goal=0.2,
                elapsed_sec=3.5,
            )
            sweep_csv = Path(tmp_dir) / "sweep_trials.csv"

            row_count = append_sweep_trial_rows(str(sweep_csv), validation_csv, "model_a", "/models/model_a")
            columns, rows = read_csv_dicts(str(sweep_csv))

        self.assertEqual(row_count, 1)
        self.assertEqual(columns[:2], ["model_name", "model_path"])
        self.assertEqual(rows[0]["model_name"], "model_a")
        self.assertEqual(rows[0]["model_path"], "/models/model_a")
        self.assertEqual(rows[0]["pair_id"], "7")
        self.assertEqual(rows[0]["planner_id"], "model_a")

    def test_sweep_summary_metrics_cover_success_collision_and_failure(self):
        rows = [
            {"reached_goal": "True", "collided": "False", "elapsed_sec": "2.0", "distance_to_goal": "0.1"},
            {"reached_goal": "False", "collided": "True", "elapsed_sec": "4.0", "distance_to_goal": "1.1"},
        ]
        summary = compute_sweep_summary(rows, "model_a", "/models/model_a")

        self.assertEqual(summary["num_trials"], 2)
        self.assertEqual(summary["reached_count"], 1)
        self.assertEqual(summary["collision_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_elapsed_sec"], 3.0)
        self.assertAlmostEqual(summary["mean_distance_to_goal"], 0.6)
        self.assertEqual(summary["status"], "ok")

        failed = compute_sweep_summary([], "model_b", "/models/model_b", status="failed", error="launch failed")
        self.assertEqual(failed["num_trials"], 0)
        self.assertEqual(failed["success_rate"], 0.0)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "launch failed")

    def test_sweep_summary_writer_uses_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_csv = Path(tmp_dir) / "sweep_summary.csv"
            append_sweep_summary_row(
                str(summary_csv),
                compute_sweep_summary([], "model_a", "/models/model_a", status="failed", error="bad model"),
            )
            columns, rows = read_csv_dicts(str(summary_csv))

        self.assertEqual(
            columns,
            [
                "model_name",
                "model_path",
                "num_trials",
                "reached_count",
                "collision_count",
                "success_rate",
                "mean_elapsed_sec",
                "mean_distance_to_goal",
                "status",
                "error",
            ],
        )
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["error"], "bad model")

    def test_launch_targets_ros1_package_and_node(self):
        launch_path = PKG_ROOT / "launch" / "planner_validation.launch"
        root = ET.fromstring(launch_path.read_text())
        nodes = root.findall("node")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].attrib["pkg"], "planner_validation_ros1")
        self.assertEqual(nodes[0].attrib["type"], "planner_validation_node.py")

    def test_cmake_installs_sweep_runner(self):
        cmake_text = (PKG_ROOT / "CMakeLists.txt").read_text()
        self.assertIn("scripts/planner_validation_node.py", cmake_text)
        self.assertIn("scripts/viplanner_model_sweep.py", cmake_text)

    def test_package_source_does_not_import_ros2_modules(self):
        banned_exact = {"rclpy", "ament_index_python", "launch", "launch_ros"}
        package_paths = list((PKG_ROOT / "src" / "planner_validation_ros1").glob("*.py"))
        package_paths.append(PKG_ROOT / "scripts" / "planner_validation_node.py")
        package_paths.append(PKG_ROOT / "scripts" / "viplanner_model_sweep.py")
        for file_path in package_paths:
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imports = [node.module]
                else:
                    continue
                for import_name in imports:
                    self.assertFalse(
                        any(import_name == banned or import_name.startswith(f"{banned}.") for banned in banned_exact),
                        f"{file_path} imports ROS2-only module {import_name}",
                    )


if __name__ == "__main__":
    unittest.main()
