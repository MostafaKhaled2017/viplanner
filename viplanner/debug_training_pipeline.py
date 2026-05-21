from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms

from viplanner.config import DataCfg, TrainCfg
from viplanner.plannernet import AutoEncoder, DualAutoEncoder, get_m2f_cfg
from viplanner.traj_cost_opt import TrajCost
from viplanner.utils.dataset import PlannerData, PlannerDataGenerator
from viplanner.utils.debug_summary import (
    array_stats,
    loss_frame_tensors,
    sanitize_for_json,
    tensor_stats,
    tensor_to_list,
    xyz_summary,
)

torch.set_default_dtype(torch.float32)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump VIPlanner training samples, model inputs, and model outputs.")
    parser.add_argument("--model-dir", required=True, help="Trained model directory containing model.pt and model.yaml.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split to inspect.")
    parser.add_argument("--num-samples", type=int, default=16, help="Maximum number of samples to dump.")
    parser.add_argument("--env-index", type=int, default=None, help="Optional env_list index to inspect.")
    parser.add_argument("--output-dir", default=None, help="Debug dump directory. Defaults to <model-dir>/debug_training.")
    parser.add_argument("--device", default=None, help="Torch device override, for example cpu or cuda:0.")
    parser.add_argument("--save-input-tensors", action="store_true", help="Save sample tensors and outputs as .npz files.")
    parser.add_argument("--allow-augmentation", action="store_true", help="Allow dataset augmentation when rebuilding splits.")
    return parser.parse_args(argv)


def _resolve_model_files(model_dir: Path) -> Tuple[Path, Path]:
    model_path = model_dir / "model.pt"
    config_path = model_dir / "model.yaml"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return model_path, config_path


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, tuple) and checkpoint and isinstance(checkpoint[0], dict):
        return checkpoint[0]
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise RuntimeError(f"Unsupported checkpoint format: {type(checkpoint)}")


def _data_cfg_for_env(cfg: TrainCfg, env_index: int) -> DataCfg:
    data_cfg = cfg.data_cfg
    if isinstance(data_cfg, list):
        return data_cfg[env_index]
    return data_cfg


def _selected_env_indices(cfg: TrainCfg, split: str, env_index: Optional[int]) -> List[int]:
    if env_index is not None:
        if env_index < 0 or env_index >= len(cfg.env_list):
            raise IndexError(f"--env-index {env_index} is outside env_list length {len(cfg.env_list)}")
        return [env_index]
    if split == "test":
        return [cfg.test_env_id]
    return [idx for idx in range(len(cfg.env_list)) if idx != cfg.test_env_id]


def _load_model(cfg: TrainCfg, model_path: Path, device: torch.device):
    if cfg.sem or cfg.rgb:
        m2f_cfg = None
        weight_path = None
        if cfg.rgb and cfg.pre_train_sem:
            pre_train_cfg = Path(cfg.all_model_dir) / str(cfg.pre_train_cfg)
            pre_train_weights = Path(cfg.all_model_dir) / str(cfg.pre_train_weights) if cfg.pre_train_weights else None
            m2f_cfg = get_m2f_cfg(str(pre_train_cfg))
            weight_path = str(pre_train_weights) if pre_train_weights else None
        model = DualAutoEncoder(cfg, m2f_cfg=m2f_cfg, weight_path=weight_path)
    else:
        model = AutoEncoder(cfg.in_channel, cfg.knodes)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def _prepare_dataset(
    cfg: TrainCfg,
    env_index: int,
    split: str,
    transform,
    traj_cost: TrajCost,
    allow_augmentation: bool,
) -> Tuple[PlannerData, PlannerDataGenerator]:
    env_name = cfg.env_list[env_index]
    root = Path(cfg.data_dir) / env_name
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset environment directory not found: {root}")

    generator = PlannerDataGenerator(
        cfg=_data_cfg_for_env(cfg, env_index),
        root=str(root),
        semantics=cfg.sem,
        rgb=cfg.rgb,
        cost_map=traj_cost.cost_map,
    )

    train_dataset = PlannerData(generator._cfg, transform=transform, semantics=cfg.sem, rgb=cfg.rgb)
    eval_dataset = PlannerData(generator._cfg, transform=transform, semantics=cfg.sem, rgb=cfg.rgb)
    generator.split_samples(
        train_dataset=train_dataset if split in {"train", "val"} else None,
        test_dataset=eval_dataset,
        generate_split=split in {"train", "val"},
        allow_augmentation=allow_augmentation,
    )
    dataset = train_dataset if split == "train" else eval_dataset
    return dataset, generator


def _loss_metrics(traj_cost: TrajCost, dataset_name: str) -> Dict[str, float]:
    metrics = traj_cost.get_loss_metrics(dataset_name)
    return {name: float(value) for name, value in metrics.items()}


def _sample_dump(
    *,
    sample_idx: int,
    dataset: PlannerData,
    model,
    traj_cost: TrajCost,
    device: torch.device,
    dataset_name: str,
    save_npz_path: Optional[Path],
    fear_ahead_dist: float,
) -> Dict[str, object]:
    depth_image, sem_rgb_image, odom, goal, augment = dataset[sample_idx]
    augment_bool = bool(augment)

    depth = depth_image.unsqueeze(0).to(device).float()
    goal_input = goal.unsqueeze(0).to(device).float()
    if isinstance(sem_rgb_image, torch.Tensor):
        sem_rgb = sem_rgb_image.unsqueeze(0).to(device).float()
    else:
        sem_rgb = None

    with torch.no_grad():
        if sem_rgb is not None:
            preds, fear = model(depth, sem_rgb, goal_input)
        else:
            preds, fear = model(depth, goal_input)

    preds_loss, goal_loss = loss_frame_tensors(preds, goal_input, augment_bool)
    traj_cost.reset_loss_metrics(dataset_name)
    loss, waypoints = traj_cost.CostofTraj(
        traj_cost.opt.TrajGeneratorFromPFreeRot(preds_loss, step=0.1),
        odom.unsqueeze(0).to(device).float(),
        goal_loss,
        fear,
        log_step=0,
        ahead_dist=fear_ahead_dist,
        dataset=dataset_name,
        context=f"debug sample {sample_idx}",
    ), traj_cost.opt.TrajGeneratorFromPFreeRot(preds_loss, step=0.1)

    dump = {
        "sample_index": int(sample_idx),
        "depth_filename": dataset.depth_filename[sample_idx],
        "sem_rgb_filename": dataset.sem_rgb_filename[sample_idx] if dataset.sem_rgb_filename else None,
        "pair_augment": augment_bool,
        "depth_tensor": tensor_stats(depth_image),
        "sem_rgb_tensor": tensor_stats(sem_rgb_image) if isinstance(sem_rgb_image, torch.Tensor) else None,
        "odom": tensor_to_list(odom),
        "goal_model_input": tensor_to_list(goal),
        "goal_loss_frame": tensor_to_list(goal_loss.squeeze(0)),
        "goal_z_abs": float(abs(goal[2].item())),
        "raw_keypoints": tensor_to_list(preds.squeeze(0)),
        "raw_keypoints_xyz": xyz_summary(preds),
        "loss_frame_keypoints": tensor_to_list(preds_loss.squeeze(0)),
        "trajectory_waypoints": tensor_to_list(waypoints.squeeze(0)),
        "trajectory_xyz": xyz_summary(waypoints),
        "fear": tensor_to_list(fear.squeeze()),
        "loss_total": float(loss.detach().cpu().item()),
        "loss_components": _loss_metrics(traj_cost, dataset_name),
    }

    if save_npz_path is not None:
        np.savez_compressed(
            save_npz_path,
            depth=depth_image.detach().cpu().numpy(),
            sem_rgb=sem_rgb_image.detach().cpu().numpy() if isinstance(sem_rgb_image, torch.Tensor) else np.array([]),
            odom=odom.detach().cpu().numpy(),
            goal=goal.detach().cpu().numpy(),
            raw_keypoints=preds.detach().cpu().numpy(),
            loss_frame_keypoints=preds_loss.detach().cpu().numpy(),
            waypoints=waypoints.detach().cpu().numpy(),
            fear=fear.detach().cpu().numpy(),
        )
        dump["npz_path"] = str(save_npz_path)

    return dump


def run_debug_dump(args: argparse.Namespace) -> Path:
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_path, config_path = _resolve_model_files(model_dir)
    cfg = TrainCfg.from_yaml(str(config_path))
    device = torch.device(args.device or (f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"))
    gpu_id = (device.index if device.index is not None else cfg.gpu_id) if device.type == "cuda" else None
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else model_dir / "debug_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(tuple(cfg.img_input_size), antialias=True)])
    model = _load_model(cfg, model_path, device)

    env_summaries = []
    for env_index in _selected_env_indices(cfg, args.split, args.env_index):
        env_name = cfg.env_list[env_index]
        traj_cost = TrajCost(
            gpu_id=gpu_id,
            log_data=True,
            w_obs=cfg.w_obs,
            w_height=cfg.w_height,
            w_goal=cfg.w_goal,
            w_motion=cfg.w_motion,
            obstalce_thread=cfg.obstacle_thread,
        )
        env_root = Path(cfg.data_dir) / env_name
        traj_cost.SetMap(str(env_root), cfg.cost_map_name)
        dataset, generator = _prepare_dataset(cfg, env_index, args.split, transform, traj_cost, args.allow_augmentation)
        try:
            sample_count = min(args.num_samples, len(dataset))
            samples = []
            for sample_idx in range(sample_count):
                npz_path = output_dir / f"{env_name}_{args.split}_{sample_idx:04d}.npz" if args.save_input_tensors else None
                samples.append(
                    _sample_dump(
                        sample_idx=sample_idx,
                        dataset=dataset,
                        model=model,
                        traj_cost=traj_cost,
                        device=device,
                        dataset_name=args.split,
                        save_npz_path=npz_path,
                        fear_ahead_dist=cfg.fear_ahead_dist,
                    )
                )
        finally:
            generator.cleanup()
        env_summaries.append(
            {
                "env_index": int(env_index),
                "env_name": env_name,
                "dataset_length": int(len(dataset)),
                "sample_count": int(sample_count),
                "goal_z_abs": array_stats([sample["goal_z_abs"] for sample in samples]),
                "samples": samples,
            }
        )

    summary = {
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "split": args.split,
        "device": str(device),
        "save_input_tensors": bool(args.save_input_tensors),
        "envs": env_summaries,
    }
    output_path = output_dir / f"debug_training_{args.split}.json"
    output_path.write_text(json.dumps(sanitize_for_json(summary), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    output_path = run_debug_dump(args)
    print(f"[INFO] Wrote training debug dump to {output_path}")


if __name__ == "__main__":
    main()
