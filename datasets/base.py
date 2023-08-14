from typing import Literal, Callable, Optional
import glob 
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm 

from ops.boxes import poly2obb_np, poly2hbb_np
from evaluation.evaluator import eval_rbbox_map
# TODO: this consumes enormous amount of memory for datasets like DOTA with large number of instances,
# need to find a way to reduce memory usage
class BaseDataset(Dataset):
    CLASSES = ()

    PALETTE = []
    
    def __init__(self, 
                 save_dir: str,
                 data_path: str,
                 angle_version: Literal["oc", "le90", "le135"]="oc", 
                 hbb_version: Literal["xyxy", "xywh"]="xyxy",
                 split: Literal["train", "test"]="train",
                 ):
        """
        Args:
            angle_version: angle version of the dataset, one of ["oc", "le90", "le135"]. Currently only "oc" is supported.
        
        """
        super(BaseDataset, self).__init__()
        self.angle_version = angle_version
        self.hbb_version = hbb_version
        self.data = self.prepare_data(save_dir, data_path).get(split)
    
    @classmethod
    def class_to_idx(cls, class_name):
        return cls.CLASSES.index(class_name) + 1 # 0 is background
    
    @classmethod
    def idx_to_class(cls, idx):
        return cls.CLASSES[idx - 1] # 0 is background
    
    @classmethod
    def get_palette(cls, value):
        if isinstance(value, str):
            value = cls.class_to_idx(value)
        return cls.PALETTE[value - 1]
    
    def prepare_data(self, save_dir, data_path):
        assert save_dir.split(".")[-1] in ("pth", "pt"), "save_dir must be a .pth or .pt file"

        if not os.path.isfile(save_dir):
            print(save_dir, "not found, creating...")
        else:
            print(save_dir, "found, loading...")
            return torch.load(save_dir)
        
        train_anns = self.load_anns(os.path.join(data_path, "trainval/annfiles"))
        test_anns = self.load_anns(os.path.join(data_path, "test/annfiles"))
        data = {
            "train": train_anns,
            "test": test_anns,
            "angle": self.angle_version,
            "hbb": self.hbb_version
        }
        torch.save(data, save_dir)
        return data
        
    def load_anns(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA annotations txt files
        """
        cls_map = {c: i+1 for i, c in enumerate(self.CLASSES)}
        cls_map['background'] = 0
        
        ann_files = glob.glob(ann_folder + '/*.txt')
        img_files = [ann_file.replace("annfiles", "images").replace(".txt", ".png") for ann_file in ann_files]

        anns = []
        image_id = 0
        for img_file, ann_file in tqdm(zip(img_files, ann_files), desc="Loading annotations", total=len(img_files)):
            gt_difficulty = []
            
            gt_bboxes = []
            gt_areas = []

            gt_obboxes = []
            gt_oareas = []
            
            gt_labels = []
            gt_polygons = []
            
            if os.path.getsize(ann_file) == 0:
                continue

            with open(ann_file, "r") as f:
                s = f.readlines()
                for si in s:
                    bbox_info = si.split()
                    
                    poly = np.array(bbox_info[:8], dtype=np.float32)
                    try:
                        obb = poly2obb_np(poly, self.angle_version)
                        hbb = poly2hbb_np(poly, self.hbb_version)
                        assert hbb[2] > hbb[0] and hbb[3] > hbb[1], "hbb must be valid"
                        
                        if not obb:
                            # Weird error in DOTA dataset, skip this instance
                            print(obb, hbb)
                            continue
                    except:  # noqa: E722
                        continue
                    
                    difficulty = int(bbox_info[9])
                    cls_name = bbox_info[8]
                    label = cls_map[cls_name]

                    gt_difficulty.append(difficulty)
                    gt_bboxes.append(hbb)
                    gt_obboxes.append(obb)
                    gt_labels.append(label)
                    gt_polygons.append(poly)
                    
                    if self.hbb_version == "xyxy":
                        w, h = hbb[2] - hbb[0], hbb[3] - hbb[1]
                        gt_areas.append(w * h)
                        
                    elif self.hbb_version == "xywh":
                        gt_areas.append(hbb[2] * hbb[3])
                    
                    gt_oareas.append(obb[2] * obb[3])
                                            
            ann = {}
            if gt_bboxes:
                # convert to array then tensor 
                # https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_new.cpp#L262 
                # may be due to memory allocation
                ann['difficulty'] = torch.tensor(np.array(gt_difficulty), dtype=torch.int64)
                ann['image_path'] = img_file
                ann['ann_path'] = ann_file
                ann['bboxes'] = torch.tensor(np.array(gt_bboxes), dtype=torch.float32)
                ann['area'] = torch.tensor(np.array(gt_areas), dtype=torch.float32)
                
                ann['oboxes'] = torch.tensor(np.array(gt_obboxes), dtype=torch.float32)
                ann['oarea'] = torch.tensor(np.array(gt_oareas), dtype=torch.float32)
                
                ann['labels'] = torch.tensor(np.array(gt_labels), dtype=torch.int64)
                ann['polygons'] = torch.tensor(np.array(gt_polygons), dtype=torch.float32)
                ann['image_id'] = torch.tensor([image_id], dtype=torch.int64)
                ann['iscrowd'] = torch.zeros((len(gt_bboxes),), dtype=torch.int64)
                
                image_id += 1
                
            else:
                continue

            anns.append(ann)
        return anns
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ann = self.data[idx]
        image = read_image(ann['image_path'], ImageReadMode.RGB) / 255.0
        return image, ann

    # Adapted from mmrotate/datasets/*.py
    def evaluate(self,
                 results,
                 metric='mAP',
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count() // 2)
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

def collate_fn(batch):
    return tuple(zip(*batch))