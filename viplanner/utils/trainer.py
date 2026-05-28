# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib

# python
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import tqdm
import yaml

# imperative-planning-learning
from viplanner.config import TrainCfg
from viplanner.plannernet import (
    PRE_TRAIN_POSSIBLE,
    AutoEncoder,
    DualAutoEncoder,
    get_m2f_cfg,
)
from viplanner.plannernet.PlannerNet import PlannerNet
from viplanner.traj_cost_opt import TrajCost, TrajViz
from viplanner.utils.torchutil import EarlyStopScheduler, count_parameters

from .dataset import PlannerData, PlannerDataGenerator

torch.set_default_dtype(torch.float32)


class Trainer:
    """
    VIPlanner Trainer
    """

    _LOSS_LOG_ORDER = (
        ("00_total_loss", None),
        ("01_height_loss", "height_loss"),
        ("02_obstacle_loss", "obstacle_loss"),
        ("03_goal_loss", "goal_loss"),
        ("04_motion_loss", "motion_loss"),
        ("05_trajectory_loss", "trajectory_loss"),
        ("06_collision_loss", "collision_loss"),
    )

    def __init__(self, cfg: TrainCfg) -> None:
        self._cfg = cfg

        # set model save/load path
        os.makedirs(self._cfg.curr_model_dir, exist_ok=True)
        self.model_path = os.path.join(self._cfg.curr_model_dir, "model.pt")
        self.checkpoint_dir = os.path.join(self._cfg.curr_model_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.config_path = os.path.join(self._cfg.curr_model_dir, "model.yaml")
        self.log_path = os.path.join(self._cfg.log_dir, self._cfg.get_model_save())
        self.log_writer: Optional[SummaryWriter] = None
        self._write_config_snapshot()
        if self._cfg.hierarchical:
            self.model_dir_hierarch = os.path.join(self._cfg.curr_model_dir, "hierarchical")
            os.makedirs(self.model_dir_hierarch, exist_ok=True)
            self.hierach_losses = {}

        # image transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self._cfg.img_input_size), antialias=True),
            ]
        )

        # init buffers DATA
        self.data_generators: List[PlannerDataGenerator] = []
        self.data_traj_cost: List[TrajCost] = []
        self.data_traj_viz: List[TrajViz] = []
        self.fov_ratio: float = None
        self.front_ratio: float = None
        self.back_ratio: float = None
        self.pixel_mean: np.ndarray = None
        self.pixel_std: np.ndarray = None

        # inti buffers MODEL
        self.best_loss = float("inf")
        self.test_loss = float("inf")
        self.net: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: EarlyStopScheduler = None

        print("[INFO] Trainer initialized")
        return

    """PUBLIC METHODS"""

    def train(self) -> None:
        print("[INFO] Start Training")
        # init logging
        self._init_logging()
        # load model and prepare model for training
        if self._cfg.resume and self._cfg.resume_model_path is None:
            raise ValueError("Resuming training requires --resume-from or TrainCfg.resume_model_path")
        self._load_model(self._cfg.resume, checkpoint_path=self._cfg.resume_model_path)
        self._configure_optimizer()

        # get dataloader for training
        self._load_data(train=True)
        if self._cfg.hierarchical:
            step_counter = 0
            train_loader_list, val_loader_list = self._get_dataloader(step=step_counter)
        else:
            train_loader_list, val_loader_list = self._get_dataloader()

        early_stop_counter = 0
        for epoch in range(self._cfg.epochs):
            train_loss = 0
            val_loss = 0
            self._reset_epoch_loss_metrics()
            for i in range(len(train_loader_list)):
                train_loss += self._train_epoch(train_loader_list[i], epoch, env_id=i)
                val_loss += self._test_epoch(val_loader_list[i], env_id=i, epoch=epoch)

            train_loss /= len(train_loader_list)
            val_loss /= len(train_loader_list)

            self._log_epoch_loss_metrics(
                dataset="train",
                total_loss=train_loss,
                component_metrics=self._collect_epoch_loss_metrics("train"),
                epoch=epoch,
            )
            self._log_epoch_loss_metrics(
                dataset="val",
                total_loss=val_loss,
                component_metrics=self._collect_epoch_loss_metrics("val"),
                epoch=epoch,
            )
            self._flush_logging()

            if val_loss < self.best_loss:
                print("[INFO] Save model of epoch %d" % (epoch))
                torch.save((self.net.state_dict(), val_loss), self.model_path)
                self.best_loss = val_loss
                early_stop_counter = 0
                print("[INFO] Current val loss: %.4f" % (self.best_loss))
            else:
                early_stop_counter += 1
                print(
                    "[INFO] Early stopping counter: "
                    f"{early_stop_counter}/{self._cfg.early_stop_patience}"
                )

            self.scheduler.step(val_loss)

            if self._cfg.checkpoint_interval > 0 and (epoch + 1) % self._cfg.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1:04d}.pt",
                )
                print(f"[INFO] Save periodic checkpoint of epoch {epoch + 1} to {checkpoint_path}")
                torch.save((self.net.state_dict(), val_loss), checkpoint_path)

            if early_stop_counter >= self._cfg.early_stop_patience:
                print("[INFO] Early stopping patience reached")
                break

            if self._cfg.hierarchical and (epoch + 1) % self._cfg.hierarchical_step == 0:
                torch.save(
                    (self.net.state_dict(), self.best_loss),
                    os.path.join(
                        self.model_dir_hierarch,
                        (
                            f"model_ep{epoch}_fov{round(self.fov_ratio, 3)}_"
                            f"front{round(self.front_ratio, 3)}_"
                            f"back{round(self.back_ratio, 3)}.pt"
                        ),
                    ),
                )
                step_counter += 1
                train_loader_list, val_loader_list = self._get_dataloader(step=step_counter)
                self.hierach_losses[epoch] = self.best_loss

        torch.cuda.empty_cache()

        # cleanup data
        for generator in self.data_generators:
            generator.cleanup()

        # empty buffers
        self.data_generators = []
        self.data_traj_cost = []
        self.data_traj_viz = []
        return

    def test(
        self,
        step: Optional[int] = None,
        show_visualizations: Optional[bool] = None,
    ) -> None:
        print("[INFO] Start Testing")
        # set random seed for reproducibility
        torch.manual_seed(self._cfg.seed)

        # define step
        if step is None and self._cfg.hierarchical:
            step = int(self._cfg.epochs / self._cfg.hierarchical_step)

        # load model
        self._load_model(resume=True)
        # get dataloader for training
        self._load_data(train=False)
        _, test_loader = self._get_dataloader(train=False, step=step)
        if show_visualizations is None:
            show_visualizations = not os.getenv("EXPERIMENT_DIRECTORY")

        self.test_loss = self._test_epoch(
            test_loader[0],
            env_id=0,
            is_visual=show_visualizations,
            fov_angle=self.data_generators[0].alpha_fov,
            dataset="test",
        )

        # cleanup data
        for generator in self.data_generators:
            generator.cleanup()

    def save_config(self) -> None:
        print(f"[INFO] val_loss: {self.best_loss:.2f}, test_loss," f"{self.test_loss:.4f}")
        print(f"[INFO] Save config and loss to {self.config_path} file")
        self._write_config_snapshot(loss={"val_loss": self.best_loss, "test_loss": self.test_loss})

        # logging
        self._close_logging()

        # plot hierarchical losses
        if self._cfg.hierarchical:
            plt.figure(figsize=(10, 10))
            plt.plot(
                list(self.hierach_losses.keys()),
                list(self.hierach_losses.values()),
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("Hierarchical Losses")
            plt.savefig(os.path.join(self.model_dir_hierarch, "hierarchical_losses.png"))
            plt.close()

        return

    """PRIVATE METHODS"""

    def _write_config_snapshot(self, loss: Optional[dict] = None) -> None:
        save_dict = {"config": self._cfg.to_dict()}
        if loss is not None:
            save_dict["loss"] = loss

        with open(self.config_path, "w") as file:
            yaml.safe_dump(save_dict, file, allow_unicode=True, default_flow_style=False)
        return

    # Helper function DATA
    def _load_data(self, train: bool = True) -> None:
        if not isinstance(self._cfg.data_cfg, list):
            self._cfg.data_cfg = [self._cfg.data_cfg] * len(self._cfg.env_list)
        assert len(self._cfg.data_cfg) == len(self._cfg.env_list), (
            "Either single DataCfg or number matching number of environments" "must be provided"
        )

        for idx, env_name in enumerate(self._cfg.env_list):
            if (train and idx == self._cfg.test_env_id) or (not train and idx != self._cfg.test_env_id):
                continue

            data_path = os.path.join(self._cfg.data_dir, env_name)

            # get trajectory cost map
            traj_cost = TrajCost(
                self._cfg.gpu_id,
                log_data=train,
                w_obs=self._cfg.w_obs,
                w_height=self._cfg.w_height,
                w_goal=self._cfg.w_goal,
                w_motion=self._cfg.w_motion,
                obstalce_thread=self._cfg.obstacle_thread,
            )
            traj_cost.SetMap(
                data_path,
                self._cfg.cost_map_name,
            )

            generator = PlannerDataGenerator(
                cfg=self._cfg.data_cfg[idx],
                root=data_path,
                semantics=self._cfg.sem,
                rgb=self._cfg.rgb,
                cost_map=traj_cost.cost_map,  # trajectory cost class
            )

            traj_viz = TrajViz(
                intrinsics=generator.K_depth,
                cam_resolution=self._cfg.img_input_size,
                camera_tilt=self._cfg.camera_tilt,
                cost_map=traj_cost.cost_map,
            )

            self.data_generators.append(generator)
            self.data_traj_cost.append(traj_cost)
            self.data_traj_viz.append(traj_viz)
            print(f"LOADED DATA FOR ENVIRONMENT: {env_name}")

        print("[INFO] LOADED ALL DATA")
        return

    # Helper function TRAINING
    def _init_logging(self) -> None:
        os.makedirs(self.log_path, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=self.log_path, max_queue=10, flush_secs=10)
        self.log_writer.add_custom_scalars(
            {
                "epoch_losses": {
                    "train": ["Multiline", [f"train/{tag}" for tag, _ in self._LOSS_LOG_ORDER]],
                    "val": ["Multiline", [f"val/{tag}" for tag, _ in self._LOSS_LOG_ORDER]],
                }
            }
        )
        print(f"[INFO] TensorBoard logs: {self.log_path}")
        return

    def _log_scalar(self, tag: str, value, step: int) -> None:
        writer = getattr(self, "log_writer", None)
        if writer is None:
            return

        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        writer.add_scalar(tag, value, step)
        return

    def _flush_logging(self) -> None:
        writer = getattr(self, "log_writer", None)
        if writer is None:
            return

        writer.flush()
        return

    def _reset_epoch_loss_metrics(self) -> None:
        for traj_cost in getattr(self, "data_traj_cost", []):
            traj_cost.reset_loss_metrics("train")
            traj_cost.reset_loss_metrics("val")
        return

    def _collect_epoch_loss_metrics(self, dataset: str) -> Dict[str, float]:
        metric_sums: Dict[str, float] = {}
        metric_count = 0

        for traj_cost in getattr(self, "data_traj_cost", []):
            metrics = traj_cost.get_loss_metrics(dataset)
            if not metrics:
                continue

            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value
            metric_count += 1

        if metric_count == 0:
            return {}

        return {name: value / metric_count for name, value in metric_sums.items()}

    def _log_epoch_loss_metrics(
        self,
        dataset: str,
        total_loss: float,
        component_metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        self._log_scalar(f"{dataset}/00_total_loss", total_loss, epoch)
        for tag_name, metric_name in self._LOSS_LOG_ORDER[1:]:
            if metric_name in component_metrics:
                self._log_scalar(f"{dataset}/{tag_name}", component_metrics[metric_name], epoch)
        return

    def _close_logging(self) -> None:
        writer = getattr(self, "log_writer", None)
        if writer is None:
            return

        with contextlib.suppress(Exception):
            writer.flush()
            writer.close()
        self.log_writer = None
        return

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str) -> str:
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return str(path)

    def _load_model(
        self,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if self._cfg.sem or self._cfg.rgb:
            if self._cfg.rgb and self._cfg.pre_train_sem:
                assert PRE_TRAIN_POSSIBLE, (
                    "Pretrained model not available since either detectron2"
                    " not installed or mask2former not found in thrid_party"
                    " folder"
                )
                pre_train_cfg = os.path.join(self._cfg.all_model_dir, self._cfg.pre_train_cfg)
                pre_train_weights = (
                    os.path.join(self._cfg.all_model_dir, self._cfg.pre_train_weights)
                    if self._cfg.pre_train_weights
                    else None
                )
                m2f_cfg = get_m2f_cfg(pre_train_cfg)
                self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN
                self.pixel_std = m2f_cfg.MODEL.PIXEL_STD
            else:
                m2f_cfg = None
                pre_train_weights = None

            self.net = DualAutoEncoder(self._cfg, m2f_cfg=m2f_cfg, weight_path=pre_train_weights)
        else:
            self.net = AutoEncoder(self._cfg.in_channel, self._cfg.knodes)

        assert torch.cuda.is_available(), "Code requires GPU"
        print(f"Available GPU list: {list(range(torch.cuda.device_count()))}")
        print(f"Running on GPU: {self._cfg.gpu_id}")
        self.net = self.net.cuda(self._cfg.gpu_id)

        if resume:
            load_path = self._resolve_checkpoint_path(checkpoint_path or self.model_path)
            model_state_dict, self.best_loss = torch.load(load_path)
            self.net.load_state_dict(model_state_dict)
            print(f"Resume train from {load_path} with loss " f"{self.best_loss}")
            if checkpoint_path is not None and os.path.abspath(load_path) != os.path.abspath(self.model_path):
                torch.save((self.net.state_dict(), self.best_loss), self.model_path)
                print(f"[INFO] Save resume checkpoint copy to {self.model_path}")

        self._apply_freeze_layers()
        print(f"[INFO] MODEL LOADED ({count_parameters(self.net)} trainable parameters)")
        return

    def _apply_freeze_layers(self) -> None:
        freeze_layers = self._cfg.freeze_layers
        if isinstance(freeze_layers, bool) or not isinstance(freeze_layers, int):
            raise ValueError(f"freeze_layers must be an integer in [0, 5], got {freeze_layers!r}")
        if freeze_layers < 0 or freeze_layers > 5:
            raise ValueError(f"freeze_layers must be an integer in [0, 5], got {freeze_layers}")
        if freeze_layers == 0:
            return

        stage_names = ("conv1", "layer1", "layer2", "layer3", "layer4")
        frozen_stages = stage_names[:freeze_layers]
        frozen_encoders = []

        for encoder_name in ("encoder", "encoder_depth", "encoder_sem"):
            encoder = getattr(self.net, encoder_name, None)
            if not isinstance(encoder, PlannerNet):
                continue

            for stage_name in frozen_stages:
                stage = getattr(encoder, stage_name)
                for param in stage.parameters():
                    param.requires_grad = False
            frozen_encoders.append(encoder_name)

        if frozen_encoders:
            print(
                "[INFO] Frozen PlannerNet encoder stages "
                f"{', '.join(frozen_stages)} for {', '.join(frozen_encoders)}"
            )
        return

    def _configure_optimizer(self) -> None:
        trainable_params = [param for param in self.net.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters remain after applying freeze_layers")

        if self._cfg.optimizer == "adam":
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self._cfg.lr,
                weight_decay=self._cfg.w_decay,
            )
        elif self._cfg.optimizer == "sgd":
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self._cfg.lr,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.w_decay,
            )
        else:
            raise KeyError(f"Optimizer {self._cfg.optimizer} not supported")
        self.scheduler = EarlyStopScheduler(
            self.optimizer,
            factor=self._cfg.factor,
            verbose=True,
            min_lr=self._cfg.min_lr,
            patience=self._cfg.patience,
        )
        print("[INFO] OPTIMIZER AND SCHEDULER CONFIGURED")
        return

    def _get_dataloader(
        self,
        train: bool = True,
        step: Optional[int] = None,
        allow_augmentation: bool = True,
    ) -> None:
        train_loader_list: List[Data.DataLoader] = []
        val_loader_list: List[Data.DataLoader] = []

        if step is not None:
            self.fov_ratio = (
                1.0 - (self._cfg.hierarchical_front_step_ratio + self._cfg.hierarchical_back_step_ratio) * step
            )
            self.front_ratio = self._cfg.hierarchical_front_step_ratio * step
            self.back_ratio = self._cfg.hierarchical_back_step_ratio * step

        for generator in self.data_generators:
            # init data classes

            val_data = PlannerData(
                cfg=generator._cfg,
                transform=self.transform,
                semantics=self._cfg.sem,
                rgb=self._cfg.rgb,
                pixel_mean=self.pixel_mean,
                pixel_std=self.pixel_std,
            )

            if train:
                train_data = PlannerData(
                    cfg=generator._cfg,
                    transform=self.transform,
                    semantics=self._cfg.sem,
                    rgb=self._cfg.rgb,
                    pixel_mean=self.pixel_mean,
                    pixel_std=self.pixel_std,
                )
            else:
                train_data = None

            # split data in train and validation with given sample ratios
            if train:
                generator.split_samples(
                    train_dataset=train_data,
                    test_dataset=val_data,
                    generate_split=train,
                    ratio_back_samples=self.back_ratio,
                    ratio_front_samples=self.front_ratio,
                    ratio_fov_samples=self.fov_ratio,
                    allow_augmentation=allow_augmentation,
                )
            else:
                generator.split_samples(
                    train_dataset=train_data,
                    test_dataset=val_data,
                    generate_split=train,
                    ratio_back_samples=self.back_ratio,
                    ratio_front_samples=self.front_ratio,
                    ratio_fov_samples=self.fov_ratio,
                    allow_augmentation=allow_augmentation,
                )

            if self._cfg.load_in_ram:
                if train:
                    train_data.load_data_in_memory()
                val_data.load_data_in_memory()

            if train:
                train_loader = Data.DataLoader(
                    dataset=train_data,
                    batch_size=self._cfg.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=self._cfg.num_workers,
                )
            val_loader = Data.DataLoader(
                dataset=val_data,
                batch_size=self._cfg.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self._cfg.num_workers,
            )

            if train:
                train_loader_list.append(train_loader)
            val_loader_list.append(val_loader)

        return train_loader_list, val_loader_list

    def _train_epoch(
        self,
        loader: Data.DataLoader,
        epoch: int,
        env_id: int,
    ) -> float:
        train_loss, batches = 0, len(loader)
        enumerater = tqdm.tqdm(enumerate(loader))

        for batch_idx, inputs in enumerater:
            odom = inputs[2].cuda(self._cfg.gpu_id)
            goal = inputs[3].cuda(self._cfg.gpu_id)
            self.optimizer.zero_grad()

            if self._cfg.sem or self._cfg.rgb:
                depth_image = inputs[0].cuda(self._cfg.gpu_id)
                sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)
                preds, fear = self.net(depth_image, sem_rgb_image, goal)
            else:
                image = inputs[0].cuda(self._cfg.gpu_id)
                preds, fear = self.net(image, goal)

            # flip y axis for augmented samples  (clone necessary due to
            # inplace operation that otherwise leads to error in backprop)
            preds_flip = torch.clone(preds)
            preds_flip[inputs[4], :, 1] = preds_flip[inputs[4], :, 1] * -1
            goal_flip = torch.clone(goal)
            goal_flip[inputs[4], 1] = goal_flip[inputs[4], 1] * -1

            loss, _ = self._loss(
                preds_flip,
                fear,
                self.data_traj_cost[env_id],
                odom,
                goal_flip,
                log_step=epoch,
                context=f"dataset=train, epoch={epoch}, env_id={env_id}, batch_idx={batch_idx}",
            )

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            enumerater.set_description(
                f"Epoch: {epoch} in Env: "
                f"({env_id+1}/{len(self._cfg.env_list)-1}) "
                f"- train loss:{round(train_loss/(batch_idx+1), 4)} on"
                f" {batch_idx}/{batches}"
            )
        return train_loss / (batch_idx + 1)

    def _test_epoch(
        self,
        loader,
        env_id: int,
        epoch: int = 0,
        is_visual=False,
        fov_angle: float = 90.0,
        dataset: str = "val",
    ) -> float:
        test_loss = 0
        num_batches = len(loader)
        preds_viz = []
        wp_viz = []
        image_viz = []

        with torch.no_grad():
            for batch_idx, inputs in enumerate(loader):
                odom = inputs[2].cuda(self._cfg.gpu_id)
                goal = inputs[3].cuda(self._cfg.gpu_id)

                if self._cfg.sem or self._cfg.rgb:
                    image = inputs[0].cuda(self._cfg.gpu_id)  # depth
                    sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)  # sem
                    preds, fear = self.net(image, sem_rgb_image, goal)
                else:
                    image = inputs[0].cuda(self._cfg.gpu_id)
                    preds, fear = self.net(image, goal)

                # flip y axis for augmented samples
                preds[inputs[4], :, 1] = preds[inputs[4], :, 1] * -1
                goal[inputs[4], 1] = goal[inputs[4], 1] * -1

                loss, waypoints = self._loss(
                    preds,
                    fear,
                    self.data_traj_cost[env_id],
                    odom,
                    goal,
                    log_step=epoch,
                    dataset=dataset,
                    context=f"dataset={dataset}, epoch={epoch}, env_id={env_id}, batch_idx={batch_idx}",
                )

                test_loss += loss.item()

                if is_visual and len(preds_viz) * batch_idx < self._cfg.n_visualize:
                    if batch_idx == 0:
                        odom_viz = odom.cpu()
                        goal_viz = goal.cpu()
                        fear_viz = fear.cpu()
                        augment_viz = inputs[4].cpu()
                    else:
                        odom_viz = torch.cat((odom_viz, odom.cpu()), dim=0)
                        goal_viz = torch.cat((goal_viz, goal.cpu()), dim=0)
                        fear_viz = torch.cat((fear_viz, fear.cpu()), dim=0)
                        augment_viz = torch.cat((augment_viz, inputs[4].cpu()), dim=0)
                    preds_viz.append(preds.cpu())
                    wp_viz.append(waypoints.cpu())
                    image_viz.append(image.cpu())

            if is_visual:
                preds_viz = torch.vstack(preds_viz)
                wp_viz = torch.vstack(wp_viz)
                image_viz = torch.vstack(image_viz)

                # limit again to number of visualizations since before
                # added as multiple of batch size
                preds_viz = preds_viz[: self._cfg.n_visualize]
                wp_viz = wp_viz[: self._cfg.n_visualize]
                image_viz = image_viz[: self._cfg.n_visualize]
                odom_viz = odom_viz[: self._cfg.n_visualize]
                goal_viz = goal_viz[: self._cfg.n_visualize]
                fear_viz = fear_viz[: self._cfg.n_visualize]
                augment_viz = augment_viz[: self._cfg.n_visualize]

                # visual trajectory and images
                self.data_traj_viz[env_id].VizTrajectory(
                    preds_viz,
                    wp_viz,
                    odom_viz,
                    goal_viz,
                    fear_viz,
                    fov_angle=fov_angle,
                    augment_viz=augment_viz,
                )
                self.data_traj_viz[env_id].VizImages(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz, image_viz)
        return test_loss / (batch_idx + 1)

    def _loss(
        self,
        preds: torch.Tensor,
        fear: torch.Tensor,
        traj_cost: TrajCost,
        odom: torch.Tensor,
        goal: torch.Tensor,
        log_step: int,
        step: float = 0.1,
        dataset: str = "train",
        context: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=step)
        loss = traj_cost.CostofTraj(
            waypoints,
            odom,
            goal,
            fear,
            log_step,
            ahead_dist=self._cfg.fear_ahead_dist,
            dataset=dataset,
            context=context,
        )
        if not torch.isfinite(loss).all().item():
            suffix = f" ({context})" if context else ""
            raise ValueError(f"Training loss contains NaN or Inf values{suffix}")

        return loss, waypoints


# EoF
