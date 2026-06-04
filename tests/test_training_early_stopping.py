import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch

from viplanner.config import TrainCfg
from viplanner.traj_cost_opt.traj_cost import TrajCost

try:
    from viplanner.utils.trainer import Trainer
except Exception as exc:
    Trainer = object
    TRAINER_IMPORT_ERROR = exc
else:
    TRAINER_IMPORT_ERROR = None


class _FakeNet:
    def state_dict(self):
        return {"weights": 1}


class _FakeScheduler:
    def __init__(self):
        self.metrics = []

    def step(self, metrics):
        self.metrics.append(metrics)


class _FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


class _EarlyStopTrainer(Trainer):
    def __init__(self, val_losses, early_stop_patience, model_path, checkpoint_interval=10):
        model_path = Path(model_path)
        self._cfg = SimpleNamespace(
            epochs=len(val_losses),
            resume=False,
            resume_model_path=None,
            hierarchical=False,
            early_stop_patience=early_stop_patience,
            checkpoint_interval=checkpoint_interval,
        )
        self._val_losses = list(val_losses)
        self._epochs_seen = []
        self.best_loss = float("inf")
        self.net = _FakeNet()
        self.model_path = str(model_path)
        checkpoint_dir = model_path.parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = str(checkpoint_dir)
        self.scheduler = _FakeScheduler()
        self.data_generators = []
        self.log_writer = None

    def _init_logging(self):
        return

    def _load_model(self, resume=False, checkpoint_path=None):
        return

    def _configure_optimizer(self):
        return

    def _load_data(self, train=True):
        return

    def _get_dataloader(self, train=True, step=None):
        return [object()], [object()]

    def _train_epoch(self, train_loader, epoch, env_id=0):
        self._epochs_seen.append(epoch)
        return 0.0

    def _test_epoch(self, val_loader, env_id=0, epoch=None, **kwargs):
        return self._val_losses[epoch]


class TrainingEarlyStoppingTest(unittest.TestCase):
    def test_train_cfg_defaults_early_stop_patience_to_ten(self):
        self.assertEqual(TrainCfg().early_stop_patience, 10)

    def test_train_cfg_defaults_checkpoint_interval_to_ten(self):
        self.assertEqual(TrainCfg().checkpoint_interval, 10)

    def test_train_cfg_loads_early_stop_patience_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "train.yaml"
            config_path.write_text(yaml.safe_dump({"config": {"early_stop_patience": 7}}))

            cfg = TrainCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.early_stop_patience, 7)

    def test_train_cfg_loads_checkpoint_interval_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "train.yaml"
            config_path.write_text(yaml.safe_dump({"config": {"checkpoint_interval": 3}}))

            cfg = TrainCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.checkpoint_interval, 3)

    @unittest.skipIf(TRAINER_IMPORT_ERROR is not None, f"Trainer import failed: {TRAINER_IMPORT_ERROR}")
    def test_training_stops_after_consecutive_non_improving_validation_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _EarlyStopTrainer(
                val_losses=[1.0, 1.0, 1.0, 0.5],
                early_stop_patience=2,
                model_path=Path(tmp_dir) / "model.pt",
            )

            trainer.train()

            self.assertEqual(trainer._epochs_seen, [0, 1, 2])
            self.assertEqual(trainer.best_loss, 1.0)
            self.assertEqual(trainer.scheduler.metrics, [1.0, 1.0, 1.0])

    @unittest.skipIf(TRAINER_IMPORT_ERROR is not None, f"Trainer import failed: {TRAINER_IMPORT_ERROR}")
    def test_training_saves_periodic_checkpoints_without_replacing_best_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pt"
            trainer = _EarlyStopTrainer(
                val_losses=[1.0] * 20,
                early_stop_patience=25,
                model_path=model_path,
                checkpoint_interval=10,
            )

            trainer.train()

            self.assertTrue(model_path.is_file())
            self.assertTrue((Path(trainer.checkpoint_dir) / "checkpoint_epoch_0010.pt").is_file())
            self.assertTrue((Path(trainer.checkpoint_dir) / "checkpoint_epoch_0020.pt").is_file())
            self.assertFalse((Path(trainer.checkpoint_dir) / "checkpoint_epoch_0009.pt").exists())

    @unittest.skipIf(TRAINER_IMPORT_ERROR is not None, f"Trainer import failed: {TRAINER_IMPORT_ERROR}")
    def test_non_positive_checkpoint_interval_disables_periodic_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _EarlyStopTrainer(
                val_losses=[1.0] * 10,
                early_stop_patience=25,
                model_path=Path(tmp_dir) / "model.pt",
                checkpoint_interval=0,
            )

            trainer.train()

            self.assertEqual(list(Path(trainer.checkpoint_dir).iterdir()), [])

    @unittest.skipIf(TRAINER_IMPORT_ERROR is not None, f"Trainer import failed: {TRAINER_IMPORT_ERROR}")
    def test_external_resume_checkpoint_resets_best_loss_for_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "source" / "model.pt"
            model_path = Path(tmp_dir) / "target" / "model.pt"
            checkpoint_path.parent.mkdir()
            model_path.parent.mkdir()
            torch.save((_FakeNet().state_dict(), 0.25), checkpoint_path)

            trainer = object.__new__(Trainer)
            trainer._cfg = SimpleNamespace(
                sem=False,
                rgb=False,
                in_channel=1,
                knodes=1,
                gpu_id=0,
                freeze_layers=0,
            )
            trainer.model_path = str(model_path)
            trainer.net = _FakeNet()
            trainer._apply_freeze_layers = lambda: None

            original_cuda_is_available = torch.cuda.is_available
            original_cuda_device_count = torch.cuda.device_count
            original_auto_encoder = Trainer._load_model.__globals__["AutoEncoder"]
            original_count_parameters = Trainer._load_model.__globals__["count_parameters"]

            class _CudaNet(_FakeNet):
                def cuda(self, gpu_id):
                    return self

                def load_state_dict(self, state_dict):
                    self.loaded_state_dict = state_dict

            try:
                torch.cuda.is_available = lambda: True
                torch.cuda.device_count = lambda: 1
                Trainer._load_model.__globals__["AutoEncoder"] = lambda in_channel, knodes: _CudaNet()
                Trainer._load_model.__globals__["count_parameters"] = lambda net: 1

                trainer._load_model(resume=True, checkpoint_path=str(checkpoint_path))
            finally:
                torch.cuda.is_available = original_cuda_is_available
                torch.cuda.device_count = original_cuda_device_count
                Trainer._load_model.__globals__["AutoEncoder"] = original_auto_encoder
                Trainer._load_model.__globals__["count_parameters"] = original_count_parameters

            self.assertEqual(trainer.best_loss, float("inf"))
            _, copied_loss = torch.load(model_path)
            self.assertEqual(copied_loss, 0.25)

    @unittest.skipIf(TRAINER_IMPORT_ERROR is not None, f"Trainer import failed: {TRAINER_IMPORT_ERROR}")
    def test_epoch_tensorboard_logs_total_loss_before_components(self):
        trainer = object.__new__(Trainer)
        trainer.log_writer = _FakeWriter()

        trainer._log_epoch_loss_metrics(
            dataset="train",
            total_loss=10.0,
            component_metrics={
                "height_loss": 1.0,
                "obstacle_loss": 2.0,
                "goal_loss": 3.0,
                "motion_loss": 4.0,
                "trajectory_loss": 5.0,
                "collision_loss": 6.0,
            },
            epoch=7,
        )

        self.assertEqual(
            [tag for tag, _, _ in trainer.log_writer.scalars],
            [
                "train/00_total_loss",
                "train/01_height_loss",
                "train/02_obstacle_loss",
                "train/03_goal_loss",
                "train/04_motion_loss",
                "train/05_trajectory_loss",
                "train/06_collision_loss",
            ],
        )

    def test_fear_probability_guard_rejects_non_finite_values_before_bce(self):
        fear = torch.tensor([[float("nan")], [0.5]])

        with self.assertRaisesRegex(ValueError, "epoch=1"):
            TrajCost._prepare_fear_probabilities(fear, context="epoch=1")

    def test_fear_probability_guard_clamps_valid_probabilities(self):
        fear = torch.tensor([[0.0], [0.5], [1.0]])

        clamped = TrajCost._prepare_fear_probabilities(fear)

        self.assertGreater(float(clamped.min()), 0.0)
        self.assertLess(float(clamped.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
