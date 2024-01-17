# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/jbwang1997/BboxToolkit/blob/56b6410d4d7b21f4dfc90afca2bd14ac10cedcd9/BboxToolkit/transforms.py
import numpy as np
import torch

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

def obb2poly_np(obboxes):
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

def obb2poly(obboxes):
    # torch.split not same as np.split
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