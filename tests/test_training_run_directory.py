import os
import tempfile
import unittest
from pathlib import Path

import yaml

from viplanner.config import TrainCfg
from viplanner.train import _load_cfg_from_args, _parse_args
from viplanner.utils.trainer import Trainer


class TrainingRunDirectoryTest(unittest.TestCase):
    def test_timestamp_model_dir_generation_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = TrainCfg(file_path=tmp_dir)

            model_dir_name = cfg.ensure_model_dir_name()

            self.assertRegex(model_dir_name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
            self.assertEqual(cfg.get_model_save(), model_dir_name)
            self.assertEqual(cfg.curr_model_dir, os.path.join(tmp_dir, "models", model_dir_name))

    def test_existing_model_dir_gets_collision_suffix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_root = Path(tmp_dir) / "models"
            model_root.mkdir()
            (model_root / "2026-05-19_05-44-00").mkdir()
            (model_root / "2026-05-19_05-44-00_01").mkdir()

            cfg = TrainCfg(file_path=tmp_dir, model_dir_name="2026-05-19_05-44-00")

            self.assertEqual(cfg.ensure_model_dir_name(), "2026-05-19_05-44-00_02")

    def test_trainer_writes_initial_yaml_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = TrainCfg(
                file_path=tmp_dir,
                model_dir_name="2026-05-19_05-44-00",
                sem=False,
                rgb=False,
            )

            trainer = Trainer(cfg)
            yaml_path = Path(trainer.config_path)

            self.assertTrue(yaml_path.is_file())
            self.assertNotIn("!!python", yaml_path.read_text())
            with yaml_path.open() as file:
                data = yaml.safe_load(file)

            self.assertEqual(data["config"]["model_dir_name"], "2026-05-19_05-44-00")
            self.assertEqual(data["config"]["sem"], False)
            self.assertNotIn("wb_project", data["config"])
            self.assertNotIn("wb_entity", data["config"])
            self.assertNotIn("wb_api_key", data["config"])
            self.assertNotIn("loss", data)

            trainer.best_loss = 1.25
            trainer.test_loss = 2.5
            trainer.save_config()
            with yaml_path.open() as file:
                data = yaml.safe_load(file)

            self.assertEqual(data["loss"], {"val_loss": 1.25, "test_loss": 2.5})

    def test_yaml_snapshot_contains_effective_resume_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "train.yaml"
            resume_path = Path(tmp_dir) / "previous" / "model.pt"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "config": {
                            "file_path": tmp_dir,
                            "model_dir_name": "2026-05-19_05-44-00",
                            "sem": False,
                            "rgb": False,
                        }
                    }
                )
            )

            args = _parse_args(["--config", str(config_path), "--resume-from", str(resume_path)])
            cfg = _load_cfg_from_args(args)
            trainer = Trainer(cfg)
            with open(trainer.config_path) as file:
                data = yaml.safe_load(file)

            self.assertTrue(data["config"]["resume"])
            self.assertEqual(data["config"]["resume_model_path"], str(resume_path))

    def test_resolve_checkpoint_accepts_file_or_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "run"
            model_dir.mkdir()
            checkpoint_path = model_dir / "model.pt"
            checkpoint_path.touch()

            self.assertEqual(Trainer._resolve_checkpoint_path(str(checkpoint_path)), str(checkpoint_path))
            self.assertEqual(Trainer._resolve_checkpoint_path(str(model_dir)), str(checkpoint_path))

    def test_yaml_loader_ignores_legacy_wandb_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "train.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "config": {
                            "file_path": tmp_dir,
                            "sem": False,
                            "rgb": False,
                            "wb_project": "legacy-project",
                            "wb_entity": "legacy-entity",
                            "wb_api_key": "legacy-key",
                        }
                    }
                )
            )

            cfg = TrainCfg.from_yaml(str(config_path))

            self.assertFalse(hasattr(cfg, "wb_project"))
            self.assertFalse(hasattr(cfg, "wb_entity"))
            self.assertFalse(hasattr(cfg, "wb_api_key"))

    def test_tensorboard_log_path_uses_timestamped_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = TrainCfg(
                file_path=tmp_dir,
                model_dir_name="2026-05-19_05-44-00",
                sem=False,
                rgb=False,
            )

            trainer = Trainer(cfg)

            self.assertEqual(
                trainer.log_path,
                os.path.join(tmp_dir, "logs", "2026-05-19_05-44-00"),
            )


if __name__ == "__main__":
    unittest.main()
