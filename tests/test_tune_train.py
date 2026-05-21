import tempfile
import unittest
from pathlib import Path

import yaml

from viplanner.tune_train import (
    GpuScheduler,
    Trial,
    _load_yaml,
    _write_trial_config,
    create_trials,
    expand_grid,
    limit_trials,
    load_sweep_config,
    parse_trial_result,
    run_trials,
)


class TuneTrainTest(unittest.TestCase):
    def test_load_sweep_config_reads_yaml_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sweep_path = Path(tmp_dir) / "sweep.yaml"
            sweep_path.write_text(
                yaml.safe_dump(
                    {
                        "base_config": "viplanner/config/train.yaml",
                        "output_root": "tuning/example",
                        "seed": 11,
                        "max_trials": 3,
                        "defaults": {"epochs": 80},
                        "fixed": {"optimizer": "sgd"},
                        "grid": {"lr": [0.001, 0.002]},
                    }
                )
            )

            spec = load_sweep_config(str(sweep_path))

            self.assertEqual(spec.base_config, Path("viplanner/config/train.yaml"))
            self.assertEqual(spec.output_root, Path("tuning/example"))
            self.assertEqual(spec.seed, 11)
            self.assertEqual(spec.max_trials, 3)
            self.assertEqual(spec.defaults, {"epochs": 80})
            self.assertEqual(spec.fixed, {"optimizer": "sgd"})
            self.assertEqual(spec.grid, {"lr": [0.001, 0.002]})

    def test_expand_grid_uses_cartesian_product(self):
        trials = expand_grid({"lr": [0.001, 0.002], "batch_size": [16, 32]})

        self.assertEqual(
            trials,
            [
                {"lr": 0.001, "batch_size": 16},
                {"lr": 0.001, "batch_size": 32},
                {"lr": 0.002, "batch_size": 16},
                {"lr": 0.002, "batch_size": 32},
            ],
        )

    def test_limit_trials_is_deterministic(self):
        trials = [{"idx": idx} for idx in range(10)]

        first = limit_trials(trials, max_trials=4, seed=7)
        second = limit_trials(trials, max_trials=4, seed=7)
        different_seed = limit_trials(trials, max_trials=4, seed=8)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotEqual(first, different_seed)

    def test_create_trials_applies_defaults_fixed_and_grid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sweep_path = Path(tmp_dir) / "sweep.yaml"
            sweep_path.write_text(
                yaml.safe_dump(
                    {
                        "base_config": "base.yaml",
                        "output_root": str(Path(tmp_dir) / "out"),
                        "defaults": {"epochs": 80, "lr": 0.001},
                        "fixed": {"optimizer": "sgd"},
                        "grid": {"lr": [0.002], "batch_size": [16, 32]},
                    }
                )
            )
            spec = load_sweep_config(str(sweep_path))

            trials = create_trials(spec)

            self.assertEqual([trial.name for trial in trials], ["trial_0000", "trial_0001"])
            self.assertEqual(
                trials[0].params,
                {"epochs": 80, "lr": 0.002, "optimizer": "sgd", "batch_size": 16},
            )
            self.assertEqual(
                trials[1].params,
                {"epochs": 80, "lr": 0.002, "optimizer": "sgd", "batch_size": 32},
            )

    def test_write_trial_config_preserves_unrelated_base_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "configs" / "trial.yaml"
            base_config = {
                "config": {
                    "file_path": "src/planner",
                    "env_list": ["env_a", "env_b"],
                    "data_cfg": {
                        "max_depth": 15.0,
                        "distance_scheme": {1: 0.2, 5: 0.8},
                    },
                }
            }
            trial = Trial(
                name="trial_0000",
                params={
                    "lr": 0.001,
                    "data_cfg.distance_scheme": {1: 0.5, 5: 0.5},
                },
                config_path=config_path,
                model_dir_name="trial_0000",
            )

            _write_trial_config(base_config, trial, trial.params, gpu_id=2)
            saved = _load_yaml(config_path)

            self.assertEqual(saved["config"]["file_path"], "src/planner")
            self.assertEqual(saved["config"]["env_list"], ["env_a", "env_b"])
            self.assertEqual(saved["config"]["data_cfg"]["max_depth"], 15.0)
            self.assertEqual(saved["config"]["data_cfg"]["distance_scheme"], {1: 0.5, 5: 0.5})
            self.assertEqual(saved["config"]["lr"], 0.001)
            self.assertEqual(saved["config"]["gpu_id"], 2)
            self.assertEqual(saved["config"]["model_dir_name"], "trial_0000")

    def test_gpu_scheduler_allows_oversubscribing_gpus(self):
        scheduler = GpuScheduler([0, 1], max_parallel=4)

        assigned_gpus = [scheduler.acquire(f"trial_{idx:04d}") for idx in range(4)]

        self.assertEqual(assigned_gpus, [0, 1, 0, 1])
        self.assertFalse(scheduler.can_start())

        scheduler.release("trial_0001")
        self.assertTrue(scheduler.can_start())
        self.assertEqual(scheduler.acquire("trial_0004"), 0)

    def test_gpu_scheduler_preserves_one_slot_per_gpu_when_parallel_matches_gpu_count(self):
        scheduler = GpuScheduler([0, 1], max_parallel=2)

        first_gpu = scheduler.acquire("trial_0000")
        second_gpu = scheduler.acquire("trial_0001")

        self.assertNotEqual(first_gpu, second_gpu)
        self.assertFalse(scheduler.can_start())

        scheduler.release("trial_0000")
        self.assertTrue(scheduler.can_start())
        self.assertEqual(scheduler.acquire("trial_0002"), first_gpu)

    def test_dry_run_writes_configs_with_round_robin_gpu_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trials = [
                Trial(
                    name=f"trial_{idx:04d}",
                    params={"lr": 0.001 + idx},
                    config_path=tmp_path / "out" / "configs" / f"trial_{idx:04d}.yaml",
                    model_dir_name=f"trial_{idx:04d}",
                )
                for idx in range(3)
            ]

            run_trials(
                trials=trials,
                base_config={"config": {"file_path": str(tmp_path)}},
                gpu_ids=[0, 1],
                max_parallel=2,
                output_root=tmp_path / "out",
                train_script=Path("viplanner/train.py"),
                show_test_visualizations=False,
                dry_run=True,
            )

            gpu_ids = [_load_yaml(trial.config_path)["config"]["gpu_id"] for trial in trials]
            self.assertEqual(gpu_ids, [0, 1, 0])
            self.assertTrue((tmp_path / "out" / "summary.yaml").is_file())
            self.assertTrue((tmp_path / "out" / "summary.csv").is_file())

    def test_parse_trial_result_reads_successful_model_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "trial.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "config": {
                            "file_path": str(tmp_path),
                            "model_dir_name": "trial_0000",
                        }
                    }
                )
            )
            model_dir = tmp_path / "models" / "trial_0000"
            model_dir.mkdir(parents=True)
            (model_dir / "model.yaml").write_text(
                yaml.safe_dump({"loss": {"val_loss": 1.25, "test_loss": 2.5}})
            )
            trial = Trial(
                name="trial_0000",
                params={"lr": 0.001},
                config_path=config_path,
                model_dir_name="trial_0000",
            )

            result = parse_trial_result(trial, return_code=0, log_path=tmp_path / "trial.log")

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["val_loss"], 1.25)
            self.assertEqual(result["test_loss"], 2.5)
            self.assertEqual(result["model_dir"], str(model_dir))
            self.assertEqual(result["model_path"], str(model_dir / "model.pt"))

    def test_parse_trial_result_marks_missing_model_yaml_failed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "trial.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "config": {
                            "file_path": str(tmp_path),
                            "model_dir_name": "trial_0000",
                        }
                    }
                )
            )
            trial = Trial(
                name="trial_0000",
                params={"lr": 0.001},
                config_path=config_path,
                model_dir_name="trial_0000",
            )

            result = parse_trial_result(trial, return_code=0, log_path=tmp_path / "trial.log")

            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["val_loss"])
            self.assertIsNone(result["test_loss"])

    def test_parse_trial_result_marks_nonzero_return_code_failed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "trial.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "config": {
                            "file_path": str(tmp_path),
                            "model_dir_name": "trial_0000",
                        }
                    }
                )
            )
            model_dir = tmp_path / "models" / "trial_0000"
            model_dir.mkdir(parents=True)
            (model_dir / "model.yaml").write_text(
                yaml.safe_dump({"loss": {"val_loss": 1.25, "test_loss": 2.5}})
            )
            trial = Trial(
                name="trial_0000",
                params={"lr": 0.001},
                config_path=config_path,
                model_dir_name="trial_0000",
            )

            result = parse_trial_result(trial, return_code=1, log_path=tmp_path / "trial.log")

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["return_code"], 1)
            self.assertEqual(result["val_loss"], 1.25)


if __name__ == "__main__":
    unittest.main()
