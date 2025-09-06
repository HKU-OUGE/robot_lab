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
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
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
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
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
    num_rows=20,
    num_cols=40,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "floating_ring": terrain_gen.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=0.3,                           # 完全生成此地形
            ring_width_range=(2.0, 2.0),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.0, 0.9),             # 环的离地高度范围
            ring_thickness=0.1,                       # 环厚度（z 方向）
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
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.45), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,   
                    patch_radius=0.15,   
                    max_height_diff=0.05
                )
            },
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.45), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,    
                    patch_radius=0.15,    
                    max_height_diff=0.05 
                )
            },
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.03), noise_step=0.005, border_width=0.25,
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
            ring_width_range=(2.5, 2.5),              # 环的宽度范围（中心向外延伸 0.5~1.0 米）
            ring_height_range=(0.20, 0.20),             # 环的离地高度范围
            ring_thickness=0.1,                       # 环厚度（z 方向）
            platform_width=5.0,                       # 地形中心的方形平台大小
            size=(10.0, 10.0),                        # 每块地形大小
        )
    }
)

PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=5.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=1.0,                     
            pit_depth_range=(0.1, 0.35),              # 坑的深度范围
            platform_width=3.0,                      # 中心平台宽度
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
"""Rough terrains configuration."""
