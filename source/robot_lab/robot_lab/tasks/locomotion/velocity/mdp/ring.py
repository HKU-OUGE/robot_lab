# --- Handstand & Height-gated reward utilities ---
import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import rewards as base_rew  # 复用内置 base_height_l2 等函数
import robot_lab.tasks.locomotion.velocity.mdp as mdp
def _reduce_ignore_nan(x: torch.Tensor, dim: int, reduce: str) -> torch.Tensor:
    """在给定 dim 上做忽略 NaN 的归约，兼容无 torch.nanmax 的 PyTorch。"""
    mask = torch.isfinite(x)
    if reduce == "max":
        fill = torch.finfo(x.dtype).min
        x2 = torch.where(mask, x, torch.tensor(fill, device=x.device, dtype=x.dtype))
        vals = torch.max(x2, dim=dim).values
        all_invalid = mask.sum(dim=dim) == 0
        return torch.where(all_invalid, torch.zeros_like(vals), vals)
    elif reduce == "min":
        fill = torch.finfo(x.dtype).max
        x2 = torch.where(mask, x, torch.tensor(fill, device=x.device, dtype=x.dtype))
        vals = torch.min(x2, dim=dim).values
        all_invalid = mask.sum(dim=dim) == 0
        return torch.where(all_invalid, torch.zeros_like(vals), vals)
    elif reduce == "mean":
        x2 = torch.where(mask, x, torch.zeros_like(x))
        cnt = mask.sum(dim=dim).clamp_min(1)
        return x2.sum(dim=dim) / cnt
    else:
        # 默认用 max
        return _reduce_ignore_nan(x, dim, "max")

def _height_gate_scalar(
    env,
    sensor_cfg: SceneEntityCfg,          # SceneEntityCfg("height_scanner")
    h_low: float = 0.10,                 # ≤10cm 强抑制（但 gate 仍为 0）
    h_start: float = 0.25,               # ≥25cm 开始启用
    h_full: float = 0.50,                # ≥50cm 全开
    use_disc: bool = True,
    reduce: str = "max",
    offset: float = 0.5,
):
    """
    计算门控系数 g∈[0,1]：h<h_start→0；h_start≤h<h_full 线性上升；h≥h_full→1。
    建议 reduce 用 'max'：见到最高的可攀平台即可触发。height_scan 会先减去 offset。 
    """
    try:
        # 首选：直接用 mdp.height_scan（官方：返回值已减 offset）
        heights = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)  # [N, B] 或 [N]
        if heights.ndim == 2:
            if reduce == "max":
                h = _reduce_ignore_nan(heights, dim=1, reduce="max")
            elif reduce == "min":
                h = _reduce_ignore_nan(heights, dim=1, reduce="min")
            else:
                h = _reduce_ignore_nan(heights, dim=1, reduce="mean")
        else:
            h = heights  # [N]
    except Exception:
        # 兜底：直接从 RayCaster 取世界系命中点 ray_hits_w[...,2] 做相对高度（同样减 offset）
        sensor = env.scene.sensors[sensor_cfg.name]
        z_sensor = sensor.data.pos_w[:, 2]            # [N]
        z_hits = sensor.data.ray_hits_w[..., 2]       # [N, B]
        valid = torch.isfinite(z_hits)
        # 无效命中放到传感器下方远处，避免干扰 max
        z_hits = torch.where(valid, z_hits, z_sensor.unsqueeze(1) - 1000.0)
        heights = z_sensor.unsqueeze(1) - z_hits - offset  # [N, B]
        h = _reduce_ignore_nan(heights, dim=1, reduce=reduce)

    # 将高度 h→门控 g：严格按 25/50 cm 起用与全开；≤h_low 也保持 0
    g = torch.zeros_like(h)
    g = torch.where(h >= h_full, torch.ones_like(g), g)
    mid = (h - h_start) / max(1e-6, (h_full - h_start))
    mid = torch.clamp(mid, 0.0, 1.0)
    g = torch.where((h >= h_start) & (h < h_full), mid, g)
    return torch.clamp(g, 0.0, 1.0)

def base_height_l2_gated(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_sensor_cfg: SceneEntityCfg | None = None,
    h_low: float = 0.10, h_start: float = 0.25, h_full: float = 0.50,
    use_disc: bool = True, offset: float = 0.5,
):
    # 原始损失（L2）——注意这是“惩罚项”，权重应为负
    base_loss = mdp.base_height_l2(env, target_height=target_height, asset_cfg=asset_cfg)
    if height_sensor_cfg is None:
        return base_loss
    g = _height_gate_scalar(
        env, height_sensor_cfg, h_low=h_low, h_start=h_start, h_full=h_full,
        use_disc=use_disc, reduce="max", offset=offset
    )
    return base_loss * g