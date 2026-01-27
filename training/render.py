# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import os
import math
import random
import numpy as np
import logging
from iopath.common.file_io import g_pathmgr
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Iterable, List
import cv2
from train_utils.general import check_and_fix_inf_nan
from vggt.utils.pose_enc import extri_intri_to_pose_encoding,pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import open3d as o3d

def project_points_and_render(points, colors, intrinsic, extrinsic, image_shape):
    """
    Args:
        points: N*3  # 输入的点云
        colors: N*3  # 相应的颜色
        intrinsic: 3*3  # 相机内参矩阵
        extrinsic: 4*4  # 相机外参矩阵
        image_shape: 图像形状，以 (高度, 宽度, 通道数) 的形式给出
    """
    # 在这里使用传入的 points 和 colors 来渲染图像
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))  # 形状为 (N, 4)

    # 计算相机坐标系中的点
    camera_coordinates = extrinsic @ points_h.T
    # camera_coordinates /= camera_coordinates[2, :]  # 3xN，归一化

    # 计算投影点
    projected_points = intrinsic @ camera_coordinates[:3, :]  # 3xN
    projected_points /= projected_points[2, :]  # 3xN，进行归一化

    # 获取投影的有效像素坐标
    u = projected_points[0, :].astype(int)
    v = projected_points[1, :].astype(int)


    # 用来存储每个像素的深度和对应的颜色
    depth_buffer = np.full(image_shape[:2], np.inf)  # 初始化深度缓冲区为无穷大
    color_buffer = np.zeros((*image_shape[:2], 3), dtype=np.float32)  # 更新颜色缓冲区

    # 确保在图像范围内的有效像素
    valid_indices = np.logical_and.reduce((
        u >= 0,
        u < image_shape[1],
        v >= 0,
        v < image_shape[0]
    ))

    valid_u = u[valid_indices]
    valid_v = v[valid_indices]

    valid_depths = camera_coordinates[2, valid_indices]  # 获取深度信息
    valid_colors = colors[valid_indices].astype(np.float32)

    for idx in range(valid_u.shape[0]):
        pixel_u = valid_u[idx]
        pixel_v = valid_v[idx]
        depth = valid_depths[idx]
        color = valid_colors[idx]

        # 如果当前深度小于当前像素的深度缓冲区值，则更新颜色和深度
        if depth < depth_buffer[pixel_v, pixel_u]:
            depth_buffer[pixel_v, pixel_u] = depth
            color_buffer[pixel_v, pixel_u] = color

    # 将颜色缓冲区的结果赋值给图像
    image = color_buffer

    return image

def compute_render(predictions,batch):

    # 真实深度值
    gt_depths = batch['depths']
    gt_depths = check_and_fix_inf_nan(gt_depths, "gt_depth")

    # 原始图像
    gt_images = batch['images']

    # 投影图像
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], gt_images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # modify
#    for key in predictions.keys():
#        if isinstance(predictions[key], torch.Tensor):
#            predictions[key] = predictions[key].detach().cpu().numpy().squeeze(0)  # remove batch dimension

    # 预测深度图
    pred_depths = predictions['depth'].clone().detach().cpu().numpy().squeeze(0)

    world_points = unproject_depth_map_to_point_map(pred_depths, predictions["extrinsic"].clone().detach().cpu().numpy().squeeze(0), predictions["intrinsic"].clone().detach().cpu().numpy().squeeze(0))
    predictions["world_points_from_depth"] = world_points

    color_images = gt_images.squeeze(0)

    all_points = []
    all_colors = []

    H = pred_depths.shape[1]
    W = pred_depths.shape[2]

    image_shape = np.array([H, W])

    for s in range(color_images.shape[0]):
        depth = pred_depths[s].squeeze()
        points = world_points[s].reshape(-1, 3)

        # Extract RGB colors from the images
        image_colors = color_images[s].permute(1, 2, 0).cpu().numpy()  # Shape (H, W, C)
        image_colors = image_colors.reshape(-1, 3)  # Flatten to shape (H*W, C)

        # Get valid points where depth is greater than 0
        valid_mask = depth > 0
        valid_indices = np.where(valid_mask.ravel())[0]  # Flatten and get valid indices

        if valid_indices.size > 0:
            all_points.append(points[valid_indices])
            all_colors.append(image_colors[valid_indices])

    # Concatenate all points and colors
    if all_points and all_colors:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

    # 获取渲染图像
    extrinsic = extrinsic.squeeze(0)
    intrinsic = intrinsic.squeeze(0)
    rendered_images = []

    for i in range(extrinsic.shape[0]):
        # 获取每个相机外参
        per_extrinsic = extrinsic[i].detach().cpu().numpy()
        per_extrinsic = np.vstack((per_extrinsic, np.array([0, 0, 0, 1])))
        per_intrinsic = intrinsic[i].detach().cpu().numpy()

        # 渲染图像
        rendered_image = project_points_and_render(all_points, all_colors, per_intrinsic, per_extrinsic, image_shape)
        rendered_images.append(rendered_image)

    rendered_images= np.array(rendered_images)

    # 可视化验证
    vis_pred_depth = pred_depths
    vis_pred_depth = vis_pred_depth.squeeze(-1)
    vis_gt_depth = gt_depths.squeeze(0)
    vis_pred_depth = vis_pred_depth[0]
    vis_gt_depth = vis_gt_depth[0].detach().cpu().numpy()
    vis_pred_depth_normalized = cv2.normalize(vis_pred_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    vis_gt_depth_normalized = cv2.normalize(vis_gt_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    vis_pred_depth_normalized = vis_pred_depth_normalized.astype(np.uint8)
    vis_gt_depth_normalized  = vis_gt_depth_normalized.astype(np.uint8)
    vis_pred_depth_colored = cv2.applyColorMap(vis_pred_depth_normalized, cv2.COLORMAP_JET)
    vis_gt_depth_colored = cv2.applyColorMap(vis_gt_depth_normalized, cv2.COLORMAP_JET)

    vis_gt_image = gt_images.squeeze(0)
    vis_gt_image = vis_gt_image[0].detach().cpu().numpy()
    vis_gt_image = vis_gt_image.transpose(1, 2, 0)
    vis_gt_image = (vis_gt_image * 255).astype(np.uint8)

    vis_render_image = rendered_images[0]
    vis_render_image = (vis_render_image * 255).astype(np.uint8)

    # cv2.imshow('Predicted Depth', vis_pred_depth_colored)
    # cv2.imshow('Ground Truth Depth', vis_gt_depth_colored)
    # cv2.imshow('GT',vis_gt_image)
    # cv2.imshow('Render Image', vis_render_image)
    # cv2.imwrite("GT.png",vis_gt_image)
    # cv2.imwrite("Render.png",vis_render_image)

    rendered_images = torch.tensor(rendered_images/255)
    rendered_images = rendered_images.permute(0, 3, 1, 2).unsqueeze(0)
    
    #import pdb;pdb.set_trace()

    return batch['depths'],batch['images'],predictions['depth'],rendered_images