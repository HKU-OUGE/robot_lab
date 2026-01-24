# verify_terrain.py
import argparse
from isaaclab.app import AppLauncher

# ==============================================================================
# 1. 启动 Isaac Sim
# ==============================================================================
parser = argparse.ArgumentParser(description="Verify terrain generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# 2. 导入模块
# ==============================================================================
import torch
import numpy as np
import trimesh
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
import isaaclab.terrains.trimesh.mesh_terrains_cfg as mesh_terrains_cfg

def verify_ring_height():
    print(">>> 开始深度验证 MeshFloatingRingTerrainCfg 高度逻辑...")
    
    REAL_THICKNESS = 1.2
    
    # [校准测试配置]
    # 源码逻辑：ring_height = range[1] - diff * (range[1] - range[0])
    # 也就是：Diff 0 -> range[1] (Start), Diff 1 -> range[0] (End)
    # 我们希望从 0.8 (Start/Easy) 降到 0.35 (End/Hard)
    # 所以应该设为 (0.35, 0.8) -> range[0]=0.35, range[1]=0.8
    TEST_HEIGHT_RANGE = (0.35, 0.8) 
    
    test_cfg = TerrainGeneratorCfg(
        size=(10.0, 10.0),
        border_width=0.0,
        num_rows=5, # 增加到 5 行以观察趋势
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "test_ring": mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
                proportion=1.0,
                ring_width_range=(0.5, 0.5),
                ring_height_range=TEST_HEIGHT_RANGE, 
                ring_thickness=REAL_THICKNESS,
                platform_width=2.0,
            )
        }
    )

    try:
        generator = TerrainGenerator(cfg=test_cfg, device="cpu")
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    print(f"\n>>> 地形参数: Height Range {TEST_HEIGHT_RANGE}, Rows=5")
    print(">>> 正在执行 Ray Casting 扫描...")
    
    mesh = generator.terrain_mesh
    # 兼容 Tensor/Numpy
    if isinstance(generator.terrain_origins, torch.Tensor):
        terrain_origins = generator.terrain_origins.cpu().numpy()
    else:
        terrain_origins = generator.terrain_origins

    def get_max_z(row_idx):
        center = terrain_origins[row_idx, 0] # Col 0
        scan_range = 4.0
        steps = 40
        x = np.linspace(center[0] - scan_range, center[0] + scan_range, steps)
        y = np.linspace(center[1] - scan_range, center[1] + scan_range, steps)
        xx, yy = np.meshgrid(x, y)
        
        ray_origins = np.column_stack([xx.flatten(), yy.flatten(), np.full_like(xx.flatten(), 10.0)])
        ray_directions = np.tile(np.array([0, 0, -1]), (len(ray_origins), 1))

        locations, _, _ = mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)
        
        if len(locations) == 0: return 0.0
        hit_zs = locations[:, 2]
        obstacle_zs = hit_zs[hit_zs > 0.1] # 过滤地面
        
        if len(obstacle_zs) == 0: return 0.0
        return np.max(obstacle_zs)

    # 扫描 起点(0)、中点(2)、终点(4)
    rows_to_scan = [0, 2, 4]
    results = []
    
    print(f"\n{'Row':<5} | {'Diff':<10} | {'Top Z (m)':<10} | {'Clearance (m)':<15}")
    print("-" * 50)
    
    for r in rows_to_scan:
        top_z = get_max_z(r)
        clearance = top_z - REAL_THICKNESS
        difficulty = r / 4.0 # 0.0 to 1.0
        results.append((r, difficulty, top_z, clearance))
        print(f"{r:<5} | {difficulty:<10.2f} | {top_z:<10.3f} | {clearance:<15.3f}")

    # 分析趋势
    c_start = results[0][3]
    c_end = results[-1][3]
    diff = c_end - c_start
    
    print("\n[诊断结论]")
    if abs(diff) < 0.2:
        print("⚠ 趋势平坦 (Flat): 高度几乎没有变化。")
    elif diff < 0:
        print("📉 趋势下降 (Descending): 难度增加，高度降低。")
        print(f"  ✅ 完美符合预期！(从 {c_start:.2f} 降到 {c_end:.2f})")
        print("  结论：请在训练配置中使用 (0.35, 0.8)")
    else:
        print("📈 趋势上升 (Ascending): 难度增加，高度升高。")
        print(f"  ❌ 与设定相反 (从 {c_start:.2f} 升到 {c_end:.2f})。")
        print("  建议：请交换 ring_height_range 的顺序。")

if __name__ == "__main__":
    verify_ring_height()
    simulation_app.close()