# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by: Tianyang TANG


"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg
# ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=False,
#     sub_terrains={
#         "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.05, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.05, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "boxes": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
#         ),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
#         ),
#         "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#         "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#     },
# )
EASY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

EASY_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.3,                     
            pit_depth_range=(0.0, 0.23),              # 坑的深度范围
            platform_width=3.0,                      # 中心平台宽度
            double_pit=True,                         # 启用双层坑（更难）
            size=(8.0, 8.0),                       # 每块子地形的大小
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "box": terrain_gen.mesh_terrains_cfg.MeshBoxTerrainCfg(
            proportion=0.3, box_height_range=(0.00, 0.23), platform_width=3.0, double_box=True, size=(8.0, 8.0), 
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

HARD1_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.8,                     
            pit_depth_range=(0.1, 0.5),              # 坑的深度范围
            platform_width=3.0,                      # 中心平台宽度
            double_pit=True,                         # 启用双层坑（更难）
            size=(8.0, 8.0),                       # 每块子地形的大小
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "box": terrain_gen.mesh_terrains_cfg.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.10, 0.5), platform_width=3.0, double_box=True, size=(8.0, 8.0), 
        )
    },
)

HARD2_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.8,                     
            pit_depth_range=(0.3, 0.8),              # 坑的深度范围
            platform_width=3.0,                      # 中心平台宽度
            double_pit=True,                         # 启用双层坑（更难）
            size=(8.0, 8.0),                       # 每块子地形的大小
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "box": terrain_gen.mesh_terrains_cfg.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.30, 0.8), platform_width=3.0, double_box=True, size=(8.0, 8.0), 
        )
    },
)

HARD3_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), 
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.8,                     
            pit_depth_range=(0.6, 1.2),              # 坑的深度范围
            platform_width=3.0,                      # 中心平台宽度
            double_pit=True,                         # 启用双层坑（更难）
            size=(8.0, 8.0),                       # 每块子地形的大小
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "box": terrain_gen.mesh_terrains_cfg.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.6, 1.2), platform_width=3.0, double_box=True, size=(8.0, 8.0), 
        )
    },
)

FINE1_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), 
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.8,                     
            pit_depth_range=(0.1, 1.0),              # 坑的深度范围
            platform_width=3.0,                      # 中心平台宽度
            double_pit=True,                         # 启用双层坑（更难）
            size=(8.0, 8.0),                       # 每块子地形的大小
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "box": terrain_gen.mesh_terrains_cfg.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.1, 1.0), platform_width=3.0, double_box=True, size=(8.0, 8.0), 
        )
    },
)

EASY_BOX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 1.5),
            step_width=2.0,
            platform_width=6.0,
            border_width=2.0,
            holes=False,
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 1.5),
        #     step_width=2.0,
        #     platform_width=6.0,
        #     border_width=2.0,
        #     holes=False,
        # ),
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.5,                     
            pit_depth_range=(0.05, 1.5),              # 坑的深度范围
            platform_width=6.0,                      # 中心平台宽度
            double_pit=True,                         # 启用双层坑（更难）
            size=(15.0, 15.0),                       # 每块子地形的大小
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.01, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=3.0, border_width=0.5
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=3.0, border_width=0.5
        # ),
    },
)

EASY_RING_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=0.3,                           # 完全生成此地形
            ring_width_range=(1.5, 1.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.0, 0.18),             # 环的离地高度范围
            ring_thickness=0.05,                       # 环厚度（z 方向）
            platform_width=3.0,                       # 地形中心的方形平台大小
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=2.0,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.10), noise_step=0.02, border_width=0.25
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

HARD1_RING_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=0.3,                           # 完全生成此地形
            ring_width_range=(1.5, 1.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.18, 0.65),             # 环的离地高度范围
            ring_thickness=0.05,                       # 环厚度（z 方向）
            platform_width=3.0,                       # 地形中心的方形平台大小
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.23, 0.7),
            step_width=2.0,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.55), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.2, 0.55), platform_width=2.0, border_width=0.25
        ),
    },
)
HARD1_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.23, 0.53),
            step_width=2.5,
            platform_width=2.5,
            border_width=2.5,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.23, 0.53),
            step_width=2.5,
            platform_width=2.5,
            border_width=2.5,
            holes=False,
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.5), platform_width=2.0, border_width=3.5
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.5), platform_width=2.0, border_width=3.5
        ),
    },
)

HARD2_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.35, 0.75),
            step_width=2.5,
            platform_width=2.5,
            border_width=2.5,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.35, 0.75),
            step_width=2.5,
            platform_width=2.5,
            border_width=2.5,
            holes=False,
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.3, 0.45), platform_width=2.0, border_width=3.5
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.3, 0.45), platform_width=2.0, border_width=3.5
        ),
    },
)


SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0, noise_range=(0.02, 0.04), noise_step=0.03, border_width=0.25
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5, slope_range=(0.0, 0.4), platform_width=2.0, border_width=1.0
        ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.5, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },
)

NOISE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.05), noise_step=0.05, border_width=0.25
        )
    },
)

STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.10, 1.0),
            step_width=1.5,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,     
                    patch_radius=0.15,    
                    max_height_diff=0.05
                )
            },
        ),
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.3,                     
            pit_depth_range=(0.1, 1.0),            
            platform_width=3.0,                
            double_pit=False,                       
            size=(10.0, 10.0),                    
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,     
                    patch_radius=0.15,   
                    max_height_diff=0.05
                )
            },
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,   
                    patch_radius=0.15,   
                    max_height_diff=0.05
                )
            },
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,    
                    patch_radius=0.15,    
                    max_height_diff=0.05 
                )
            },
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.05), noise_step=0.005, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,      
                    patch_radius=0.15,    
                    max_height_diff=0.05
                )
            },
        )
    },
)

PLANE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(20.0, 10.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            size=(20.0, 10.0)
        ),
    },
)


FLOATING_RING_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),       # 整个 terrain tile 尺寸
    border_width=20.0,        # 地形边界，防止掉落
    num_rows=12,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=0.3,                           # 完全生成此地形
            ring_width_range=(2.0, 2.0),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.1, 0.7),             # 环的离地高度范围
            ring_thickness=0.01,                       # 环厚度（z 方向）
            platform_width=3.0,                       # 地形中心的方形平台大小
            size=(5.0, 5.0),                        # 每块地形大小
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,     
                    patch_radius=0.15,    
                    max_height_diff=0.05
                )
            },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.10, 0.7),
            step_width=1.5,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,     
                    patch_radius=0.15,    
                    max_height_diff=0.05
                )
            },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.10, 0.7),
            step_width=1.5,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,     
                    patch_radius=0.15,    
                    max_height_diff=0.05
                )
            },
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,   
                    patch_radius=0.15,   
                    max_height_diff=0.05
                )
            },
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,    
                    patch_radius=0.15,    
                    max_height_diff=0.05 
                )
            },
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.05), noise_step=0.005, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,      
                    patch_radius=0.15,    
                    max_height_diff=0.05
                )
            },
        )
    }
)

AVOID1_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),       # 整个 terrain tile 尺寸
    border_width=20.0,        # 地形边界，防止掉落
    num_rows=12,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1.0,                           # 完全生成此地形
            ring_width_range=(3.0, 3.0),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.7, 0.35),             # 环的离地高度范围
            ring_thickness=0.1,                       # 环厚度（z 方向）
            platform_width=3.0,                       # 地形中心的方形平台大小
            size=(10.0, 10.0),                        # 每块地形大小
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,     
                    patch_radius=0.15,    
                    max_height_diff=0.05
                )
            },
        )

    }
)


MOE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),       # 整个 terrain tile 尺寸
    border_width=20.0,        # 地形边界，防止掉落
    num_rows=10,
    num_cols=14,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1/7,                           # 完全生成此地形
            ring_width_range=(0.1, 1.0),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.35, 0.8),             # 环的离地高度范围
            ring_thickness=1.2,                       # 环厚度（z 方向）
            platform_width=6.0,                       # 地形中心的方形平台大小
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,     
            #         patch_radius=0.15,    
            #         max_height_diff=0.05
            #     )
            # },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1/7,
            step_height_range=(0.05, 0.25),
            step_width=0.4,
            platform_width=2,
            border_width=1.5,
            holes=False,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,     
            #         patch_radius=0.15,    
            #         max_height_diff=0.05
            #     )
            # },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1/7,
            step_height_range=(0.05, 0.25),
            step_width=0.4,
            platform_width=2,
            border_width=1.5,
            holes=False,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,     
            #         patch_radius=0.15,    
            #         max_height_diff=0.05
            #     )
            # },
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1/7, slope_range=(0.05, 0.4), platform_width=2.0, border_width=0.25,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,   
            #         patch_radius=0.15,   
            #         max_height_diff=0.05
            #     )
            # },
        ),
        # "noisy_slope": terrain_gen.HfNoisyPyramidSlopedTerrainCfg(
        #     proportion=0.05,
        #     slope_range=(0.01, 0.4),     # 自己调
        #     platform_width=2.0,
        #     noise_range=(-0.05, 0.0),    # 在高度上加最多 2cm 的波动
        # ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1/7, slope_range=(0.05, 0.4), platform_width=2.0, border_width=0.25,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,    
            #         patch_radius=0.15,    
            #         max_height_diff=0.05 
            #     )
            # },
        ),
        "gap": terrain_gen.trimesh.mesh_terrains_cfg.MeshGapTerrainCfg(
            proportion=1/7, gap_width_range=(0.10, 0.30), platform_width=4.0,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,    
            #         patch_radius=0.15,    
            #         max_height_diff=0.05 
            #     )
            # },
        ),
        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #     proportion=1/7,
        #     stone_height_max=0.1,         # 石头高度浮动 (0.1m)
        #     stone_width_range=(0.4, 0.8), # 石头宽度 (0.4m - 0.8m)
        #     stone_distance_range=(0.1, 0.4), # 石头间距/缝隙宽度 (0.2m - 0.5m)
        #     holes_depth=-3.0,             # 坑深 (-3.0m)
        #     platform_width=4.0,
        #     horizontal_scale=0.1,         # 确保精度
        #     vertical_scale=0.005,
        # ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=1/7, rail_thickness_range=(0.05, 0.05), rail_height_range=(0.05, 0.25),platform_width=4.0,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,    
            #         patch_radius=0.15,    
            #         max_height_diff=0.05 
            #     )
            # },
        ),
        # "plane": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.10,size=(10.0, 10.0)
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.10, noise_range=(0.01, 0.05), noise_step=0.005, border_width=0.25,
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,      
        #             patch_radius=0.15,    
        #             max_height_diff=0.05
        #         )
        #     },
        # )
    }
)


GAP_RING_PRETRAIN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),       # 整个 terrain tile 尺寸
    border_width=20.0,        # 地形边界，防止掉落
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "gap": terrain_gen.trimesh.mesh_terrains_cfg.MeshGapTerrainCfg(
            proportion=1/2, gap_width_range=(0.05, 0.20), platform_width=6.0,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,    
            #         patch_radius=0.15,    
            #         max_height_diff=0.05 
            #     )
            # },
        ),
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1/2,                           # 完全生成此地形
            ring_width_range=(0.1, 0.1),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.35, 0.8),             # 环的离地高度范围
            ring_thickness=1.2,                       # 环厚度（z 方向）
            platform_width=6.0,                       # 地形中心的方形平台大小
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(
            #         num_patches=50,     
            #         patch_radius=0.15,    
            #         max_height_diff=0.05
            #     )
            # },
        ),
    }
)



FLOATING_RING_TEST_TERRAINS_CFG2 = TerrainGeneratorCfg(
    size=(9.0, 9.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,  # 确保这里是 5，对应下面 5 种地形
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
        #     proportion=0.2,  # 修改为 0.2 (1/5)
        #     ring_width_range=(2.5, 2.5),
        #     ring_height_range=(0.5, 0.5),
        #     ring_thickness=0.2,
        #     platform_width=2.0,
        # ),
        "pyramid_stairs_inv": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,  # 修改为 0.2
            step_height_range=(0.2, 0.2),
            step_width=0.4,
            platform_width=2,
            holes=False,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,  # 修改为 0.2
            step_height_range=(0.2, 0.2),
            step_width=0.4,
            platform_width=2,
            holes=False,
            border_width=0.25,
        ),
        "gap": terrain_gen.trimesh.mesh_terrains_cfg.MeshGapTerrainCfg(
            proportion=0.2,  # 修改为 0.2
            gap_width_range=(0.25, 0.25), 
            platform_width=2.0,
        ),
        "rail": terrain_gen.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=0.2,  # 修改为 0.2
            rail_thickness_range=(0.05, 0.05), 
            rail_height_range=(0.2, 0.2),
            platform_width=2.0,
        ),
    }
)


FLOATING_CAR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),       # 整个 terrain tile 尺寸
    border_width=20.0,        # 地形边界，防止掉落
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=1.0,                           # 完全生成此地形
            ring_width_range=(1.5, 1.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.45, 0.45),             # 环的离地高度范围
            ring_thickness=0.05,                       # 环厚度（z 方向）
            platform_width=5.0,                       # 地形中心的方形平台大小
            size=(10.0, 10.0),                        # 每块地形大小
        )
    }
)

PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=2.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0,                     
            pit_depth_range=(0.0, 0.95),              # 坑的深度范围
            platform_width=4.0,                      # 中心平台宽度
            double_pit=False,                         # 启用双层坑（更难）
            size=(10.0, 10.0),                       # 每块子地形的大小
        ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.25,
        #     step_height_range=(0.10, 0.5),
        #     step_width=2.0,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.25,
        #     step_height_range=(0.10, 0.5),
        #     step_width=2.0,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },

)


SAND_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),       # 整个 terrain tile 尺寸
    border_width=20.0,        # 地形边界，防止掉落
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7, noise_range=(0.00, 0.05), noise_step=0.01
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),

    }
)

SAND_SLOPE_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),       # 整个 terrain tile 尺寸
    border_width=20.0,        # 地形边界，防止掉落
    num_rows=12,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.00, 0.03), noise_step=0.01
        ),
        "noisy_slope": terrain_gen.HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.8,
            slope_range=(0.1, 0.8),     # 自己调
            platform_width=1.0,
            noise_range=(-0.05, 0.0),    # 在高度上加最多 2cm 的波动
        ),

    }
)
"""Rough terrains configuration."""
