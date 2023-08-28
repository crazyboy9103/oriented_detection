# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from mmrotate/core/evaluation/eval_map.py
from multiprocessing import get_context

import numpy as np
import torch
from terminaltables import AsciiTable

from ops.boxes import box_iou_rotated

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """

    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
            
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
        
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    
    ap = ap[0]
    return ap

def tpfp_default(det_bboxes,
                 gt_bboxes,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    det_bboxes = np.array(det_bboxes)
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    ious_to_return = np.zeros((num_scales, num_dets), dtype=np.float32)
    det_angles_to_return = np.zeros((num_scales, num_dets), dtype=np.float32)
    gt_angles_to_return = np.zeros((num_scales, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp, ious_to_return, det_angles_to_return, gt_angles_to_return

    if num_dets == 0:
        return tp, fp, ious_to_return, det_angles_to_return, gt_angles_to_return
        
    
    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes[:, :5]),
        torch.from_numpy(gt_bboxes)
    ).numpy()

    det_angles = det_bboxes[:, 4]
    gt_angles = gt_bboxes[:, 4]
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                ious_to_return[k, i] = ious_max[i]
                det_angles_to_return[k, i] = det_angles[i]
                gt_angles_to_return[k, i] = gt_angles[matched_gt]
                
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[k, i] = 1
                    
                else:
                    fp[k, i] = 1
                    
            elif min_area is None:
                fp[k, i] = 1
                ious_to_return[k, i] = ious_max[i]
                
            else:
                area = det_bboxes[i][2] * det_bboxes[i][3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
                    ious_to_return[k, i] = ious_max[i]
    return tp, fp, ious_to_return, det_angles_to_return, gt_angles_to_return


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_dets = [det for det in cls_dets if det]
    # cls_dets = list(filter(lambda x: len(x), cls_dets))
    
    cls_gts = []
    for ann in annotations:
        gt_inds = np.array(ann['labels']) == class_id + 1
        cls_gts.append(np.array(ann['oboxes'])[gt_inds, :]) 


    return cls_dets, cls_gts


def eval_rbbox_map(det_results,
                   annotations,
                   iou_thr=0.5,
                   use_07_metric=True,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `oboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    if len(det_results) == 0:
        return 0, []
    
    num_imgs = len(det_results)
    num_scales = 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = (None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for class_idx in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts = get_cls_results(det_results, annotations, class_idx)
        if not cls_dets or not cls_gts:
            continue
        # compute tp and fp for each image with multiple processes
        data = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        
        tp, fp, ious, det_angles, gt_angles = tuple(zip(*data))
        # calculate gt number of each scale
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        ious = np.hstack(ious)[:, sort_inds]
        
        nonzero_ious = ious[ious != 0].copy()
        nonzero_ious = np.where(nonzero_ious == -1, nonzero_ious + 1, nonzero_ious)

        det_angles = np.hstack(det_angles)[:, sort_inds]
        gt_angles = np.hstack(gt_angles)[:, sort_inds]
        
        nonzero_det_angles = det_angles[ious != 0].copy()
        nonzero_gt_angles = gt_angles[ious != 0].copy()
        
        abs_diff_angle = np.abs(nonzero_det_angles - nonzero_gt_angles)
        # 𝑑(𝐼𝑜𝑈,𝜃_𝑔𝑡, 𝜃_𝑑𝑡)=𝐼𝑜𝑈/(1+ln(|𝜃_𝑔𝑡−𝜃_𝑑𝑡 |+1))
        modulated_ious = nonzero_ious / (1 + np.log(abs_diff_angle+1))
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        
        recalls = recalls[0, :]
        precisions = precisions[0, :]
        num_gts = num_gts.item()
        
        mod_miou = np.mean(modulated_ious)
        miou = np.mean(nonzero_ious)
        if np.isnan(mod_miou):
            mod_miou = 0
        
        if np.isnan(miou):
            miou = 0
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap,
            'miou': miou, 
            'mod_miou': mod_miou
        })
    pool.close()
    
        
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1
        
    num_classes = len(results)

    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    mious = np.zeros((num_scales, num_classes), dtype=np.float32)
    mod_mious = np.zeros((num_scales, num_classes), dtype=np.float32)
    for i, cls_result in enumerate(results):
        if cls_result['precision'].size > 0:
            precisions[:, i] = np.array(cls_result['precision'], ndmin=2)[:, -1]
            
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']
        mious[:, i] = cls_result['miou']
        mod_mious[:, i] = cls_result['mod_miou']

    label_names = [str(i) for i in range(num_classes)]

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'precision', 'ap', 'miou', 'mod_miou', 'mod_miou/miou']
    for i in range(num_scales):
        table_data = [header]
        fracs = np.nan_to_num(mod_mious / mious)
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}',  f'{precisions[i, j]:.3f}',  f'{aps[i, j]:.3f}', f'{mious[i, j]:.3f}', f'{mod_mious[i, j]:.3f}', f'{fracs[i, j]:.3f}'
            ]
            table_data.append(row_data)
        miou = float(np.mean(mious, 1)[0])
        mmiou = float(np.mean(mod_mious, 1)[0])
        frac = float(np.mean(fracs, 1)[0])
        table_data.append(['mAP|mIoU|mmIoU|mmIoU/mIoU', '', '', '', '', f'{mean_ap[i]:.3f}', f'{miou:.3f}', f'{mmiou:.3f}', f'{frac:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print(table.table)