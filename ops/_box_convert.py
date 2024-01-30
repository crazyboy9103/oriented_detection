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

def obb2poly_np(obboxes):
    center, w, h, theta = np.split(obboxes, [2, 3, 4], axis=-1)
    theta = np.deg2rad(theta)
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
    theta = torch.deg2rad(theta)
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