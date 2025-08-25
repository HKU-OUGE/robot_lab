"""
Utility functions for converting depth images into 3D point clouds.

This module provides three helper functions that wrap Isaac Lab's math utilities
to convert camera depth images into 3D point clouds.  These functions
demonstrate how to:

* Unproject a single-channel depth map into a set of 3D points in the
  **camera** coordinate frame.
* Transform those camera–frame points into the **world** coordinate frame
  given the camera's pose (position and quaternion).
* Perform both steps in a single call for convenience.

The code assumes that you have a configured ``Camera`` sensor in your
environment which provides depth images via the ``distance_to_image_plane``
annotator and exposes intrinsic matrices as
``camera.data.intrinsic_matrices`` as well as the camera's world position and
orientation via ``camera.data.pos_w`` and ``camera.data.quat_w_ros``.

Examples
--------

Given a depth image ``depth`` of shape ``(H, W)`` or ``(H, W, 1)`` and the
intrinsic matrix ``intr`` of shape ``(3, 3)``, you can obtain a point cloud
in camera coordinates as follows:

>>> from depth_projection import depth_to_camera_points
>>> points_cam = depth_to_camera_points(depth, intr)

Assuming the camera pose ``pos_w`` and quaternion ``quat_w`` are available
from your sensor, you can then transform those points into world
coordinates:

>>> from depth_projection import camera_points_to_world
>>> points_world = camera_points_to_world(points_cam, pos_w, quat_w)

Alternatively, call ``depth_to_world_pointcloud`` directly to combine both
steps:

>>> from depth_projection import depth_to_world_pointcloud
>>> points_world = depth_to_world_pointcloud(depth, intr, pos_w, quat_w)

Notes
-----

* The functions rely on ``isaaclab.utils.math.unproject_depth`` and
  ``isaaclab.utils.math.transform_points`` to perform the heavy lifting.
  These functions are vectorized and run on the same device as the input
  tensors (e.g. CPU or GPU).
* ``unproject_depth`` expects orthogonal depth images (i.e. distances
  measured from the image plane). Since the Isaac Lab camera annotator
  ``distance_to_image_plane`` returns orthogonal depths【320852010168426†L601-L615】, this is usually the
  correct setting.  If you pass perspective depths measured from the
  camera's optical centre, set ``is_ortho=False`` to internally convert the
  depths.
* The returned point clouds have shape ``(P, 3)`` where ``P`` equals the
  number of valid depth pixels. Invalid or zero depths will be skipped.

Copyright (c) 2024-2025 Tianyang TANG
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import unproject_depth, transform_points

def depth_to_camera_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    is_ortho: bool = True,
) -> torch.Tensor:
    """Unproject a depth image into 3D points in the camera frame.

    Parameters
    ----------
    depth : torch.Tensor
        Depth image returned by the camera's ``distance_to_image_plane`` or
        ``depth`` annotator.  Shape can be ``(H, W)``, ``(H, W, 1)``,
        ``(N, H, W)`` or ``(N, H, W, 1)`` where ``N`` is the batch size.
    intrinsics : torch.Tensor
        Camera calibration matrix (intrinsics).  Shape is ``(3, 3)`` or
        ``(N, 3, 3)``.  For a single camera, pass a single ``(3, 3)`` matrix.
    is_ortho : bool, optional
        Whether the depth values represent orthogonal depth (measured from the
        image plane) or perspective depth (measured from the optical centre).
        Depths from Isaac Lab's ``distance_to_image_plane`` annotator are
        orthogonal【320852010168426†L601-L615】, so the default is ``True``.

    Returns
    -------
    torch.Tensor
        A tensor of 3D points in the camera coordinate frame.  Shape is
        ``(P, 3)`` or ``(N, P, 3)`` depending on the input shapes.  Points
        corresponding to zero or invalid depth values are filtered out.
    """
    # Isaac Lab's unproject_depth accepts depth images of shape (H, W) or
    # (H, W, 1) or batched versions and returns a flattened list of points.
    points_cam = unproject_depth(depth, intrinsics, is_ortho=is_ortho)
    return points_cam


def camera_points_to_world(
    points: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
) -> torch.Tensor:
    """Transform camera-frame points into world-frame points.

    Parameters
    ----------
    points : torch.Tensor
        3D points in the camera coordinate frame.  Shape is ``(P, 3)`` or
        ``(N, P, 3)``.
    pos : torch.Tensor
        Position of the camera in world coordinates.  Shape is ``(3,)`` or
        ``(N, 3)``.
    quat : torch.Tensor
        Quaternion of the camera orientation in world coordinates, in
        (w, x, y, z) format.  Shape is ``(4,)`` or ``(N, 4)``.

    Returns
    -------
    torch.Tensor
        The input points transformed into the world coordinate frame.  Shape
        matches the input shape ``points``.
    """
    return transform_points(points, pos=pos, quat=quat)


def depth_to_world_pointcloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    *,
    is_ortho: bool = True,
) -> torch.Tensor:
    """Convert a depth image directly into a world-frame 3D point cloud.

    This helper wraps ``depth_to_camera_points`` and ``camera_points_to_world``
    to provide a single call that unprojects the depth image and applies the
    camera pose in world space.

    Parameters
    ----------
    depth : torch.Tensor
        Depth image from the camera.  Shape ``(H, W)`` or ``(H, W, 1)`` or
        batched versions.
    intrinsics : torch.Tensor
        Camera calibration matrix.  Shape ``(3, 3)`` or ``(N, 3, 3)``.
    pos : torch.Tensor
        Position of the camera in world coordinates.  Shape ``(3,)`` or
        ``(N, 3)``.
    quat : torch.Tensor
        Orientation of the camera in world coordinates as a quaternion
        (w, x, y, z).  Shape ``(4,)`` or ``(N, 4)``.
    is_ortho : bool, optional
        Whether the depth values are measured from the image plane.  Defaults
        to ``True`` for Isaac Lab's ``distance_to_image_plane`` depths【320852010168426†L601-L615】.

    Returns
    -------
    torch.Tensor
        World-frame 3D points corresponding to the non-zero depth pixels.
        Shape is ``(P, 3)`` or ``(N, P, 3)``.
    """
    # First unproject depth into camera-frame points
    points_cam = depth_to_camera_points(depth, intrinsics, is_ortho=is_ortho)
    # Then transform those points into world-frame coordinates
    points_world = camera_points_to_world(points_cam, pos=pos, quat=quat)
    return points_world