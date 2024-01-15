# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/jbwang1997/BboxToolkit/blob/56b6410d4d7b21f4dfc90afca2bd14ac10cedcd9/BboxToolkit/transforms.py
# import cv2
import numpy as np
import torch

def poly2obb_np(polys):
    theta = np.arctan2(-(polys[..., 3] - polys[..., 1]),
                       polys[..., 2] - polys[..., 0])
    Cos, Sin = np.cos(theta), np.sin(theta)
    Matrix = np.stack([Cos, -Sin, Sin, Cos], axis=-1)
    Matrix = Matrix.reshape(*Matrix.shape[:-1], 2, 2)

    x = polys[..., 0::2].mean(-1)
    y = polys[..., 1::2].mean(-1)
    center = np.expand_dims(np.stack([x, y], axis=-1), -2)
    center_polys = polys.reshape(*polys.shape[:-1], 4, 2) - center
    rotate_polys = np.matmul(center_polys, Matrix.swapaxes(-1, -2))

    xmin = np.min(rotate_polys[..., :, 0], axis=-1)
    xmax = np.max(rotate_polys[..., :, 0], axis=-1)
    ymin = np.min(rotate_polys[..., :, 1], axis=-1)
    ymax = np.max(rotate_polys[..., :, 1], axis=-1)
    w = xmax - xmin
    h = ymax - ymin

    obboxes = np.stack([x, y, w, h, theta], axis=-1)
    return obboxes

# def poly2obb_np(poly):
#     """Convert polygons to oriented bounding boxes.

#     Args:
#         polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

#     Returns:
#         obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
#     """
#     bboxps = np.array(poly).reshape((4, 2))
#     rbbox = cv2.minAreaRect(bboxps)
#     cx, cy, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
#     if w < 2 or h < 2:
#         return
#     while not 0 < a <= 90:
#         if a == -90:
#             a += 180
#         else:
#             a += 90
#             w, h = h, w
#     a = a / 180 * np.pi
#     assert 0 < a <= np.pi / 2
#     return cx, cy, w, h, a

def poly2obb(polys):
    theta = torch.atan2(-(polys[..., 3] - polys[..., 1]),
                        polys[..., 2] - polys[..., 0])
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    Matrix = torch.stack([Cos, -Sin, Sin, Cos], dim=-1)
    Matrix = Matrix.reshape(*Matrix.shape[:-1], 2, 2)

    x = polys[..., 0::2].mean(dim=-1)
    y = polys[..., 1::2].mean(dim=-1)
    center = torch.stack([x, y], dim=-1).unsqueeze(-2)
    center_polys = polys.reshape(*polys.shape[:-1], 4, 2) - center
    rotate_polys = torch.matmul(center_polys, Matrix.transpose(-1, -2))

    xmin = torch.min(rotate_polys[..., :, 0], -1).values
    xmax = torch.max(rotate_polys[..., :, 0], -1).values
    ymin = torch.min(rotate_polys[..., :, 1], -1).values
    ymax = torch.max(rotate_polys[..., :, 1], -1).values
    w = xmax - xmin
    h = ymax - ymin

    obboxes = torch.cat([x, y, w, h, theta], dim=1)
    # print("poly2obb", obboxes.shape)
    return obboxes
# def poly2obb(polys):
#     """Convert polygons to oriented bounding boxes.

#     Args:
#         polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

#     Returns:
#         obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
#     """
#     points = torch.reshape(polys, [-1, 4, 2])
#     cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
#     cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
#     _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
#     _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
#     _thetas = torch.unsqueeze(
#         torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
#                     points[:, 1, 1] - points[:, 0, 1]),
#         axis=1)
#     odd = torch.eq(torch.remainder((_thetas / (torch.pi * 0.5)).floor_(), 2), 0)
#     ws = torch.where(odd, _hs, _ws)
#     hs = torch.where(odd, _ws, _hs)
#     thetas = torch.remainder(_thetas, torch.pi * 0.5)
#     rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
#     return rbboxes

def obb2poly(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)

    vector1 = torch.cat([w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat([-h/2 * Sin, -h/2 * Cos], dim=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    temp = torch.cat([point1, point2, point3, point4], dim=-1)
    return temp

# def obb2poly(rboxes):
#     """Convert oriented bounding boxes to polygons.

#     Args:
#         obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

#     Returns:
#         polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
#     """
#     x = rboxes[:, 0]
#     y = rboxes[:, 1]
#     w = rboxes[:, 2]
#     h = rboxes[:, 3]
#     a = rboxes[:, 4]
#     cosa = torch.cos(a)
#     sina = torch.sin(a)
#     wx, wy = w / 2 * cosa, w / 2 * sina
#     hx, hy = -h / 2 * sina, h / 2 * cosa
#     p1x, p1y = x - wx - hx, y - wy - hy
#     p2x, p2y = x + wx - hx, y + wy - hy
#     p3x, p3y = x + wx + hx, y + wy + hy
#     p4x, p4y = x - wx + hx, y - wy + hy
#     temp = torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)
#     print("temp", temp.shape)
#     return temp

def poly2hbb_np(polys):
    if polys.shape[-1] == 8:
        polys = polys.reshape(4, 2)
        
    x1, y1 = polys.min(axis=0)
    x2, y2 = polys.max(axis=0)
    return np.array([x1, y1, x2, y2])

def hbb2obb(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    is_list = isinstance(hbboxes, list)
    split_chunk = len(hbboxes)
    if is_list:
        hbboxes = torch.cat(hbboxes, dim=0)
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    results = torch.stack([x, y, w, h, theta], dim=1)
    return torch.split(results, len(results) // split_chunk, dim=0) if is_list else results

def obb2xyxy(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        0 < angle <= pi/2, so cos(angle)>0, sin(angle)>0

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    # rbboxes = rbboxes.reshape(-1, 5)
    # print("rbboxes", rbboxes.shape)
    # poly = obb2poly(rbboxes)
    # print("poly", poly.shape)
    # x1, y1 = poly[:, 0::2].min(axis=1)
    # x2, y2 = poly[:, 1::2].max(axis=1)
    # temp = torch.stack((x1, y1, x2, y2), -1)
    # print("temp", temp.shape)  
    # return temp
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)