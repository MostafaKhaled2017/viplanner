import importlib.util
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("open3d") is not None,
    "open3d is required for projection debug tests",
)
class TestDebugProjectCloudToImages(unittest.TestCase):
    def test_project_identity_pose_to_principal_point(self):
        from viplanner.debug_project_cloud_to_images import project_points_to_image

        K = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        points = np.array([[1.0, 0.0, 0.0]])

        projection = project_points_to_image(points, pose, K, (20, 20))

        self.assertEqual(len(projection.pixels), 1)
        np.testing.assert_array_equal(projection.pixels[0], np.array([10, 10]))
        np.testing.assert_allclose(projection.depths, np.array([1.0]))
        np.testing.assert_array_equal(projection.source_indices, np.array([0]))

    def test_project_filters_points_behind_camera_and_outside_image(self):
        from viplanner.debug_project_cloud_to_images import project_points_to_image

        K = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        )

        projection = project_points_to_image(points, pose, K, (20, 20))

        self.assertEqual(len(projection.pixels), 1)
        np.testing.assert_array_equal(projection.pixels[0], np.array([10, 10]))

    def test_load_depth_image_scales_and_zeros_nonfinite_values(self):
        from viplanner.debug_project_cloud_to_images import load_depth_image

        with tempfile.TemporaryDirectory() as tmp_dir:
            env_dir = Path(tmp_dir) / "forest"
            depth_dir = env_dir / "depth"
            depth_dir.mkdir(parents=True)
            np.save(
                depth_dir / "0000_cam0.npy",
                np.array([[1000.0, np.inf], [np.nan, 2000.0]], dtype=np.float32),
            )

            image = load_depth_image(str(env_dir), 0, "_cam0", 1000.0)

        np.testing.assert_allclose(image, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    def test_zbuffer_projection_keeps_nearest_point_color_and_source_index(self):
        from viplanner.debug_project_cloud_to_images import ProjectionResult, zbuffer_projection

        projection = ProjectionResult(
            pixels=np.array([[3, 4], [3, 4], [5, 4]]),
            depths=np.array([5.0, 2.0, 4.0]),
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            mask=np.ones(3, dtype=bool),
            source_indices=np.array([10, 11, 12]),
        )

        zbuffered = zbuffer_projection(projection, (10, 10))

        self.assertEqual(len(zbuffered.pixels), 2)
        kept = {tuple(pixel): (depth, tuple(color), source_idx) for pixel, depth, color, source_idx in zip(
            zbuffered.pixels,
            zbuffered.depths,
            zbuffered.colors,
            zbuffered.source_indices,
        )}
        self.assertEqual(kept[(3, 4)], (2.0, (0, 255, 0), 11))
        self.assertEqual(kept[(5, 4)], (4.0, (0, 0, 255), 12))

    def test_depth_residual_masks_classifies_close_mismatch_and_missing_depth(self):
        from viplanner.debug_project_cloud_to_images import ProjectionResult, depth_residual_masks

        depth_image = np.array([[1.0, 1.2, 0.0]], dtype=np.float32)
        projection = ProjectionResult(
            pixels=np.array([[0, 0], [1, 0], [2, 0]]),
            depths=np.array([1.05, 2.0, 1.0]),
            colors=np.zeros((3, 3), dtype=np.uint8),
            mask=np.ones(3, dtype=bool),
        )

        valid, close, residual = depth_residual_masks(depth_image, projection, residual_threshold=0.25)

        np.testing.assert_array_equal(valid, np.array([True, True, False]))
        np.testing.assert_array_equal(close, np.array([True, False, False]))
        np.testing.assert_allclose(residual[:2], np.array([0.05, 0.8]), atol=1e-6)
        self.assertTrue(np.isnan(residual[2]))

    def test_backproject_depth_to_world_matches_reconstruction_camera_convention(self):
        from viplanner.debug_project_cloud_to_images import backproject_depth_to_world

        K = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        depth = np.zeros((11, 11), dtype=np.float32)
        depth[5, 5] = 2.0

        points = backproject_depth_to_world(depth, pose, K, stride=1)

        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_allclose(points[0], np.array([2.0, 0.0, 0.0]))

    def test_zbuffer_depth_residual_uses_visible_point(self):
        from viplanner.debug_project_cloud_to_images import ProjectionResult, depth_residual_stats, zbuffer_projection

        depth_image = np.ones((10, 10), dtype=np.float32) * 2.0
        projection = ProjectionResult(
            pixels=np.array([[3, 4], [3, 4]]),
            depths=np.array([2.0, 25.0]),
            colors=np.zeros((2, 3), dtype=np.uint8),
            mask=np.ones(2, dtype=bool),
        )

        raw_stats = depth_residual_stats(depth_image, projection, prefix="raw")
        zbuffer_stats = depth_residual_stats(depth_image, zbuffer_projection(projection, (10, 10)), prefix="zbuffer")

        self.assertGreater(raw_stats["raw_depth_abs_error_mean"], 10.0)
        self.assertEqual(zbuffer_stats["zbuffer_depth_abs_error_mean"], 0.0)
        self.assertEqual(zbuffer_stats["zbuffer_depth_close_025_ratio"], 1.0)

    def test_draw_projection_alpha_blends_points_without_touching_other_pixels(self):
        from viplanner.debug_project_cloud_to_images import ProjectionResult, draw_projection

        image = np.ones((5, 5, 3), dtype=np.uint8) * 100
        projection = ProjectionResult(
            pixels=np.array([[2, 2]]),
            depths=np.array([1.0]),
            colors=np.array([[200, 50, 0]], dtype=np.uint8),
            mask=np.ones(1, dtype=bool),
        )

        opaque = draw_projection(image, projection, point_size=0, alpha=1.0)
        blended = draw_projection(image, projection, point_size=0, alpha=0.5)

        np.testing.assert_array_equal(opaque[2, 2], np.array([200, 50, 0], dtype=np.uint8))
        np.testing.assert_array_equal(blended[2, 2], np.array([150, 75, 50], dtype=np.uint8))
        np.testing.assert_array_equal(blended[0, 0], image[0, 0])

    def test_draw_projection_rejects_invalid_alpha(self):
        from viplanner.debug_project_cloud_to_images import ProjectionResult, draw_projection

        image = np.zeros((3, 3, 3), dtype=np.uint8)
        projection = ProjectionResult(
            pixels=np.zeros((0, 2), dtype=int),
            depths=np.zeros(0),
            colors=np.zeros((0, 3), dtype=np.uint8),
            mask=np.zeros(0, dtype=bool),
        )

        with self.assertRaisesRegex(ValueError, "overlay-alpha"):
            draw_projection(image, projection, point_size=1, alpha=1.5)

    def test_parse_alpha_values_requires_five_valid_values(self):
        from viplanner.debug_project_cloud_to_images import parse_alpha_values

        self.assertEqual(parse_alpha_values("0.1,0.2,0.3,0.4,0.5"), [0.1, 0.2, 0.3, 0.4, 0.5])
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            parse_alpha_values("0.1,0.2")
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            parse_alpha_values("0.1,0.2,0.3,0.4,1.2")

    def test_viewer_frame_geometries_builds_expected_debug_layers(self):
        from viplanner.config import ReconstructionCfg
        from viplanner.debug_project_cloud_to_images import DebugContext, viewer_frame_geometries

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_dir = root / "forest"
            (env_dir / "depth").mkdir(parents=True)
            (env_dir / "rgb").mkdir()

            depth = np.ones((10, 10), dtype=np.float32)
            depth[5, 3] = 0.0
            np.save(env_dir / "depth" / "0000_cam0.npy", depth)
            rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(env_dir / "rgb" / "0000_cam1.png"), rgb))

            K = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
            extrinsic = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            context = DebugContext(
                cfg=ReconstructionCfg(
                    data_dir=str(root),
                    env="forest",
                    depth_suffix="_cam0",
                    sem_suffix="_cam1",
                    depth_scale=1.0,
                    semantics=False,
                ),
                env_path=str(env_dir),
                rgb_dir=str(env_dir / "rgb"),
                output_dir=str(env_dir / "debug_projection"),
                depth_scale=1.0,
                K_depth=K,
                K_rgb=K,
                depth_extrinsics=extrinsic,
                rgb_extrinsics=extrinsic,
                points=np.array([[1.0, 0.0, 0.0], [2.0, 0.1, 0.0], [1.0, 0.2, 0.0]]),
                colors=None,
                total_cloud_points=3,
                frame_ids=[0],
                overlay_alpha=0.5,
                overlay_alpha_values=[0.0, 0.2, 0.4, 0.6, 0.75],
            )

            geometries, stats = viewer_frame_geometries(
                context,
                frame_idx=0,
                residual_threshold=0.25,
                depth_sample_stride=2,
            )

        self.assertEqual(stats["visible_points"], 3)
        self.assertEqual(stats["matched_points"], 1)
        self.assertEqual(stats["mismatch_points"], 1)
        self.assertEqual(stats["missing_depth_points"], 1)
        self.assertIn("frustum", geometries)
        self.assertEqual(np.asarray(geometries["matched"].points).shape[0], 1)
        self.assertEqual(np.asarray(geometries["mismatch"].points).shape[0], 1)
        self.assertEqual(np.asarray(geometries["missing_depth"].points).shape[0], 1)

    def test_cli_writes_overlays_and_metrics_for_synthetic_environment(self):
        import open3d as o3d

        from viplanner import debug_project_cloud_to_images

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_dir = root / "forest"
            (env_dir / "depth").mkdir(parents=True)
            (env_dir / "rgb").mkdir()
            output_dir = env_dir / "debug_projection"

            config_path = root / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    reconstruction:
                      data_dir: {root}
                      env: forest
                      depth_suffix: _cam0
                      sem_suffix: _cam1
                      depth_scale: 1000
                      semantics: false
                    """
                )
            )

            P_depth = np.array([10.0, 0.0, 5.0, 0.0, 0.0, 10.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            P_rgb = P_depth.copy()
            np.savetxt(env_dir / "intrinsics.txt", np.vstack([P_depth, P_rgb]), delimiter=",")
            extrinsic = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            np.savetxt(env_dir / "camera_extrinsic_cam0.txt", extrinsic, delimiter=",")
            np.savetxt(env_dir / "camera_extrinsic_cam1.txt", extrinsic, delimiter=",")

            depth_mm = np.ones((10, 10), dtype=np.uint16) * 1000
            self.assertTrue(cv2.imwrite(str(env_dir / "depth" / "0000_cam0.png"), depth_mm))
            rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            rgb[:, :, 1] = 255
            self.assertTrue(cv2.imwrite(str(env_dir / "rgb" / "0000_cam1.png"), rgb))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]]))
            self.assertTrue(o3d.io.write_point_cloud(str(env_dir / "cloud.ply"), pcd))

            result = debug_project_cloud_to_images.main(
                [
                    "--config",
                    str(config_path),
                    "--max-images",
                    "1",
                    "--output-dir",
                    str(output_dir),
                    "--overlay-alpha",
                    "0.5",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "rgb_overlay_0000.png").is_file())
            self.assertTrue((output_dir / "depth_overlay_0000.png").is_file())
            self.assertTrue((output_dir / "rgb_zbuffer_overlay_0000.png").is_file())
            self.assertTrue((output_dir / "depth_zbuffer_overlay_0000.png").is_file())
            self.assertTrue((output_dir / "rgb_overlay_0000_alpha_020.png").is_file())
            self.assertTrue((output_dir / "rgb_overlay_0000_alpha_075.png").is_file())
            self.assertTrue((output_dir / "depth_zbuffer_overlay_0000_alpha_020.png").is_file())
            self.assertTrue((output_dir / "depth_zbuffer_overlay_0000_alpha_075.png").is_file())
            self.assertEqual(len(list(output_dir.glob("*_alpha_*.png"))), 20)
            metrics = (output_dir / "projection_metrics.csv").read_text()
            self.assertIn("frame_idx,total_cloud_points", metrics)
            self.assertIn("zbuffer_depth_abs_error_mean", metrics)
            self.assertIn("zbuffer_depth_close_025_ratio", metrics)
            self.assertIn("0,2,2", metrics)

    def test_cli_viewer_mode_can_skip_overlay_images(self):
        import open3d as o3d

        from viplanner import debug_project_cloud_to_images

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_dir = root / "forest"
            (env_dir / "depth").mkdir(parents=True)
            (env_dir / "rgb").mkdir()
            output_dir = env_dir / "debug_projection"

            config_path = root / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    reconstruction:
                      data_dir: {root}
                      env: forest
                      depth_suffix: _cam0
                      sem_suffix: _cam1
                      depth_scale: 1000
                      semantics: false
                    """
                )
            )

            P_depth = np.array([10.0, 0.0, 5.0, 0.0, 0.0, 10.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            np.savetxt(env_dir / "intrinsics.txt", np.vstack([P_depth, P_depth]), delimiter=",")
            extrinsic = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            np.savetxt(env_dir / "camera_extrinsic_cam0.txt", extrinsic, delimiter=",")
            np.savetxt(env_dir / "camera_extrinsic_cam1.txt", extrinsic, delimiter=",")

            depth_mm = np.ones((10, 10), dtype=np.uint16) * 1000
            self.assertTrue(cv2.imwrite(str(env_dir / "depth" / "0000_cam0.png"), depth_mm))
            rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(env_dir / "rgb" / "0000_cam1.png"), rgb))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]))
            self.assertTrue(o3d.io.write_point_cloud(str(env_dir / "cloud.ply"), pcd))

            with mock.patch.object(debug_project_cloud_to_images, "run_viewer") as run_viewer:
                result = debug_project_cloud_to_images.main(
                    [
                        "--config",
                        str(config_path),
                        "--max-images",
                        "1",
                        "--output-dir",
                        str(output_dir),
                        "--viewer",
                        "--no-save-images",
                    ]
                )

            self.assertEqual(result, 0)
            run_viewer.assert_called_once()
            self.assertTrue((output_dir / "projection_metrics.csv").is_file())
            self.assertEqual(list(output_dir.glob("*.png")), [])

    def test_run_viewer_reports_when_open3d_window_creation_fails(self):
        import argparse

        from viplanner.config import ReconstructionCfg
        from viplanner import debug_project_cloud_to_images

        context = debug_project_cloud_to_images.DebugContext(
            cfg=ReconstructionCfg(),
            env_path="/tmp",
            rgb_dir="/tmp",
            output_dir="/tmp",
            depth_scale=1.0,
            K_depth=np.eye(3),
            K_rgb=np.eye(3),
            depth_extrinsics=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            rgb_extrinsics=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            points=np.empty((0, 3)),
            colors=None,
            total_cloud_points=0,
            frame_ids=[0],
            overlay_alpha=0.5,
            overlay_alpha_values=[0.0, 0.2, 0.4, 0.6, 0.75],
        )
        args = argparse.Namespace(viewer_point_size=2.0, residual_threshold=0.25, depth_sample_stride=8)

        visualizer = mock.Mock()
        visualizer.create_window.return_value = False
        with mock.patch.object(
            debug_project_cloud_to_images.o3d.visualization,
            "VisualizerWithKeyCallback",
            return_value=visualizer,
        ):
            with self.assertRaisesRegex(RuntimeError, "Open3D could not create a visible window"):
                debug_project_cloud_to_images.run_viewer(context, args)


if __name__ == "__main__":
    unittest.main()
