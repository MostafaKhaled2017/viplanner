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
    distance3,
    init_results_csv,
    load_scenarios_txt,
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

    def test_launch_targets_ros1_package_and_node(self):
        launch_path = PKG_ROOT / "launch" / "planner_validation.launch"
        root = ET.fromstring(launch_path.read_text())
        nodes = root.findall("node")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].attrib["pkg"], "planner_validation_ros1")
        self.assertEqual(nodes[0].attrib["type"], "planner_validation_node.py")

    def test_package_source_does_not_import_ros2_modules(self):
        banned_exact = {"rclpy", "ament_index_python", "launch", "launch_ros"}
        package_paths = list((PKG_ROOT / "src" / "planner_validation_ros1").glob("*.py"))
        package_paths.append(PKG_ROOT / "scripts" / "planner_validation_node.py")
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
