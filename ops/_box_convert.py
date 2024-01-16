# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/jbwang1997/BboxToolkit/blob/56b6410d4d7b21f4dfc90afca2bd14ac10cedcd9/BboxToolkit/transforms.py
# import cv2
import numpy as np
import torch

# def poly2obb_np(polys):
#     theta = np.arctan2(-(polys[..., 3] - polys[..., 1]),
#                        polys[..., 2] - polys[..., 0])
#     Cos, Sin = np.cos(theta), np.sin(theta)
#     Matrix = np.stack([Cos, -Sin, Sin, Cos], axis=-1)
#     Matrix = Matrix.reshape(*Matrix.shape[:-1], 2, 2)

#     x = polys[..., 0::2].mean(-1)
#     y = polys[..., 1::2].mean(-1)
#     center = np.expand_dims(np.stack([x, y], axis=-1), -2)
#     center_polys = polys.reshape(*polys.shape[:-1], 4, 2) - center
#     rotate_polys = np.matmul(center_polys, Matrix.swapaxes(-1, -2))

#     xmin = np.min(rotate_polys[..., :, 0], axis=-1)
#     xmax = np.max(rotate_polys[..., :, 0], axis=-1)
#     ymin = np.min(rotate_polys[..., :, 1], axis=-1)
#     ymax = np.max(rotate_polys[..., :, 1], axis=-1)
#     w = xmax - xmin
#     h = ymax - ymin
    
#     # dw = torch.abs(cosa * w/2) + torch.abs(sina * h/2)
#     # dh = torch.abs(sina * w/2) + torch.abs(cosa * h/2)

#     xywha = np.stack([x, y, w, h, theta], axis=-1)
#     return xywha

def poly2obb_np(polys):
    # (x1, y1, x2, y2, x3, y3, x4, y4)
    # -(x2-x1)/(y2-y1) = tan(theta)
    # Calculate rotation angle
    theta = np.arctan2(-(polys[..., 3] - polys[..., 1]),
                       polys[..., 2] - polys[..., 0])
    # Center of the bounding box
    x = polys[..., 0::2].mean(-1)
    y = polys[..., 1::2].mean(-1)
    
    # Calculate oriented width and height
    w = np.sqrt((polys[..., 0] - polys[..., 2])**2 + (polys[..., 1] - polys[..., 3])**2)
    h = np.sqrt((polys[..., 2] - polys[..., 4])**2 + (polys[..., 3] - polys[..., 5])**2)
    
    obb = np.stack([x, y, w, h, theta], axis=-1)
    return obb

# def poly2obb(polys):
#     theta = torch.atan2(-(polys[..., 3] - polys[..., 1]),
#                         polys[..., 2] - polys[..., 0])
#     Cos, Sin = torch.cos(theta), torch.sin(theta)
#     Matrix = torch.stack([Cos, -Sin, Sin, Cos], dim=-1)
#     Matrix = Matrix.reshape(*Matrix.shape[:-1], 2, 2)

#     x = polys[..., 0::2].mean(dim=-1)
#     y = polys[..., 1::2].mean(dim=-1)
#     center = torch.stack([x, y], dim=-1).unsqueeze(-2)
#     center_polys = polys.reshape(*polys.shape[:-1], 4, 2) - center
#     rotate_polys = torch.matmul(center_polys, Matrix.transpose(-1, -2))

#     xmin = torch.min(rotate_polys[..., :, 0], -1).values
#     xmax = torch.max(rotate_polys[..., :, 0], -1).values
#     ymin = torch.min(rotate_polys[..., :, 1], -1).values
#     ymax = torch.max(rotate_polys[..., :, 1], -1).values
#     w = xmax - xmin
#     h = ymax - ymin

#     xywha = torch.cat([x, y, w, h, theta], dim=1)
#     return xywha

def poly2obb(polys):
    # Calculate rotation angle
    theta = torch.atan2(-(polys[..., 3] - polys[..., 1]),
                        polys[..., 2] - polys[..., 0])
    # Center of the bounding box
    x = polys[..., 0::2].mean(dim=-1)
    y = polys[..., 1::2].mean(dim=-1)
    
    # Calculate oriented width and height
    w = torch.sqrt((polys[..., 0] - polys[..., 2])**2 + (polys[..., 1] - polys[..., 3])**2)
    h = torch.sqrt((polys[..., 2] - polys[..., 4])**2 + (polys[..., 3] - polys[..., 5])**2)
    
    obb = torch.stack([x, y, w, h, theta], dim=-1)
    return obb

def obb2poly(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)

    vector1 = torch.cat([w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat([-h/2 * Sin, -h/2 * Cos], dim=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    poly = torch.cat([point1, point2, point3, point4], dim=-1)
    return poly

def obb2poly_np(obboxes):
    # torch.split not same as np.split
    center, w, h, theta = np.split(obboxes, [2, 3, 4], axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w/2 * Cos, -w/2 * Sin], axis=-1)
    vector2 = np.concatenate([-h/2 * Sin, -h/2 * Cos], axis=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    poly = np.concatenate([point1, point2, point3, point4], axis=-1)
    return poly

def poly2hbb_np(polys):
    assert polys.shape[-1] == 8
    polys = polys.reshape(4, 2)
        
    x1, y1 = polys.min(axis=0)
    x2, y2 = polys.max(axis=0)
    hbb = np.array([x1, y1, x2, y2])
    return hbb

def hbb2obb(hbb):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    is_list = isinstance(hbb, list)
    if is_list:
        hbb = torch.cat(hbb, dim=0)
        
    x = (hbb[..., 0] + hbb[..., 2]) * 0.5
    y = (hbb[..., 1] + hbb[..., 3]) * 0.5
    w = hbb[..., 2] - hbb[..., 0]
    h = hbb[..., 3] - hbb[..., 1]
    theta = torch.full(x.shape, 0.0, dtype=x.dtype, device=x.device)
    results = torch.stack([x, y, w, h, theta], dim=1)
    obb = torch.split(results, dim=0) if is_list else results
    return obb

def obb2hbb(obb):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    x = obb[:, 0::5]
    y = obb[:, 1::5]
    w = obb[:, 2::5]
    h = obb[:, 3::5]
    a = obb[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    
    dw = torch.abs(cosa * w/2) + torch.abs(sina * h/2)
    dh = torch.abs(sina * w/2) + torch.abs(cosa * h/2)
    x1 = x - dw 
    y1 = y - dh 
    x2 = x + dw 
    y2 = y + dh 
    xyxy = torch.cat([x1, y1, x2, y2], dim=-1)
    return xyxy

def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)