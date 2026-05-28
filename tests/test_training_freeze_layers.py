import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from viplanner.config import TrainCfg
from viplanner.plannernet import AutoEncoder, DualAutoEncoder
from viplanner.utils.trainer import Trainer

PKG_ROOT = Path(__file__).resolve().parents[1] / "src" / "viplanner_ros1"
sys.path.insert(0, str(PKG_ROOT / "src"))
from viplanner_ros1.learning_cfg import TrainCfg as Ros1TrainCfg  # noqa: E402


def _all_trainable(module):
    return all(param.requires_grad for param in module.parameters())


def _none_trainable(module):
    return all(not param.requires_grad for param in module.parameters())


def _trainer_with_model(freeze_layers):
    cfg = TrainCfg(sem=True, rgb=False, freeze_layers=freeze_layers)
    trainer = object.__new__(Trainer)
    trainer._cfg = cfg
    trainer.net = DualAutoEncoder(cfg)
    return trainer


def _trainer_with_autoencoder(freeze_layers):
    cfg = TrainCfg(sem=False, rgb=False, freeze_layers=freeze_layers)
    trainer = object.__new__(Trainer)
    trainer._cfg = cfg
    trainer.net = AutoEncoder(cfg.in_channel, cfg.knodes)
    return trainer


class TrainingFreezeLayersTest(unittest.TestCase):
    def test_train_cfg_defaults_freeze_layers_to_zero(self):
        self.assertEqual(TrainCfg().freeze_layers, 0)

    def test_train_cfg_loads_freeze_layers_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "train.yaml"
            config_path.write_text(yaml.safe_dump({"config": {"freeze_layers": 4}}))

            cfg = TrainCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.freeze_layers, 4)

    def test_ros1_train_cfg_accepts_freeze_layers_from_saved_model_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "model.yaml"
            config_path.write_text(yaml.safe_dump({"config": {"freeze_layers": 4}}))

            cfg = Ros1TrainCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.freeze_layers, 4)

    def test_freeze_layers_zero_leaves_all_parameters_trainable(self):
        trainer = _trainer_with_model(freeze_layers=0)

        trainer._apply_freeze_layers()

        self.assertTrue(_all_trainable(trainer.net))

    def test_freeze_layers_four_freezes_early_stages_for_both_plannernet_encoders(self):
        trainer = _trainer_with_model(freeze_layers=4)

        trainer._apply_freeze_layers()

        for encoder in (trainer.net.encoder_depth, trainer.net.encoder_sem):
            self.assertTrue(_none_trainable(encoder.conv1))
            self.assertTrue(_none_trainable(encoder.layer1))
            self.assertTrue(_none_trainable(encoder.layer2))
            self.assertTrue(_none_trainable(encoder.layer3))
            self.assertTrue(_all_trainable(encoder.layer4))
        self.assertTrue(_all_trainable(trainer.net.decoder))

    def test_freeze_layers_five_freezes_all_encoder_stages_but_keeps_decoder_trainable(self):
        trainer = _trainer_with_model(freeze_layers=5)

        trainer._apply_freeze_layers()

        for encoder in (trainer.net.encoder_depth, trainer.net.encoder_sem):
            self.assertTrue(_none_trainable(encoder.conv1))
            self.assertTrue(_none_trainable(encoder.layer1))
            self.assertTrue(_none_trainable(encoder.layer2))
            self.assertTrue(_none_trainable(encoder.layer3))
            self.assertTrue(_none_trainable(encoder.layer4))
        self.assertTrue(_all_trainable(trainer.net.decoder))

    def test_freeze_layers_apply_to_single_stream_autoencoder(self):
        trainer = _trainer_with_autoencoder(freeze_layers=2)

        trainer._apply_freeze_layers()

        self.assertTrue(_none_trainable(trainer.net.encoder.conv1))
        self.assertTrue(_none_trainable(trainer.net.encoder.layer1))
        self.assertTrue(_all_trainable(trainer.net.encoder.layer2))
        self.assertTrue(_all_trainable(trainer.net.decoder))

    def test_invalid_freeze_layers_values_raise(self):
        for freeze_layers in (-1, 6, True, "4"):
            trainer = _trainer_with_model(freeze_layers=0)
            trainer._cfg.freeze_layers = freeze_layers
            with self.subTest(freeze_layers=freeze_layers):
                with self.assertRaises(ValueError):
                    trainer._apply_freeze_layers()

    def test_optimizer_contains_only_trainable_parameters_after_freezing(self):
        trainer = _trainer_with_model(freeze_layers=4)
        trainer._apply_freeze_layers()

        trainer._configure_optimizer()

        optimizer_params = {id(param) for group in trainer.optimizer.param_groups for param in group["params"]}
        trainable_params = {id(param) for param in trainer.net.parameters() if param.requires_grad}
        self.assertEqual(optimizer_params, trainable_params)
        self.assertTrue(all(param.requires_grad for group in trainer.optimizer.param_groups for param in group["params"]))


if __name__ == "__main__":
    unittest.main()
