import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from viplanner.config import TrainCfg

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
    def __init__(self, val_losses, early_stop_patience, model_path):
        self._cfg = SimpleNamespace(
            epochs=len(val_losses),
            resume=False,
            resume_model_path=None,
            hierarchical=False,
            early_stop_patience=early_stop_patience,
        )
        self._val_losses = list(val_losses)
        self._epochs_seen = []
        self.best_loss = float("inf")
        self.net = _FakeNet()
        self.model_path = str(model_path)
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

    def test_train_cfg_loads_early_stop_patience_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "train.yaml"
            config_path.write_text(yaml.safe_dump({"config": {"early_stop_patience": 7}}))

            cfg = TrainCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.early_stop_patience, 7)

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


if __name__ == "__main__":
    unittest.main()
