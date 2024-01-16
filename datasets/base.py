from typing import Literal, Tuple
import glob 
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm 

from ops import boxes as box_ops
# TODO: this consumes enormous amount of memory for datasets like DOTA with large number of instances,
# need to find a way to reduce memory usage
class BaseDataset(Dataset):
    CLASSES = ()

    PALETTE = []
    
    IMAGE_MEAN: Tuple[float, float, float] 
    IMAGE_STD: Tuple[float, float, float]
    
    def __init__(self, 
                 save_dir: str,
                 data_path: str,
                 split: Literal["train", "test"]="train",
                 ):
        """
        Args:
            save_dir (str): pth path to save the processed data
            data_path (str): path to the dataset folder
            split (str): one of ["train", "test"]    
        """
        super(BaseDataset, self).__init__()
        self.data = self.prepare_data(save_dir, data_path).get(split)
    
    @classmethod
    def class_to_idx(cls, class_name):
        return cls.CLASSES.index(class_name) # 0 is background
    
    @classmethod
    def idx_to_class(cls, idx):
        return cls.CLASSES[idx] # 0 is background
    
    @classmethod
    def get_palette(cls, value):
        if isinstance(value, str):
            value = cls.class_to_idx(value)
        return cls.PALETTE[value]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ann = self.data[idx]
        image = read_image(ann['image_path'], ImageReadMode.RGB).float() / 255.0
        return image, ann
    
    def prepare_data(self, save_dir, data_path):
        assert save_dir.split(".")[-1] in ("pth", "pt"), "save_dir must be a .pth or .pt file"

        if not os.path.isfile(save_dir):
            print("[DATA]", save_dir, "not found, creating...")
        else:
            print("[DATA]", save_dir, "found, loading...")
            return torch.load(save_dir)
        
        train_anns = self.load_anns(os.path.join(data_path, "trainval/annfiles"))
        test_anns = self.load_anns(os.path.join(data_path, "test/annfiles"))
        data = {
            "train": train_anns,
            "test": test_anns,
        }
        torch.save(data, save_dir)
        return data
    
    def load_anns(self, ann_folder):
        raise NotImplementedError
    
    
class BaseXY4Dataset(BaseDataset):
    def load_anns(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA annotations txt files
        """
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        
        ann_files = glob.glob(ann_folder + '/*.txt')
        # Use images with annotations
        img_files = [ann_file.replace("annfiles", "images").replace(".txt", ".png") for ann_file in ann_files]
        # # For det_demo
        # if not os.path.isfile(img_files[0]):
        #     img_files = [ann_file.replace("annfiles", "images").replace(".txt", ".jpg") for ann_file in ann_files]
        #     assert img_files, "No image files found"
            
        anns = []
        image_id = 0
        for img_file, ann_file in tqdm(zip(img_files, ann_files), desc="Loading annotations", total=len(img_files)):
            gt_difficulty = []
            
            gt_bboxes = []
            # gt_areas = []

            gt_oboxes = []
            # gt_oareas = []
            
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
                        obb = box_ops.poly2obb_np(poly)
                        hbb = box_ops.poly2hbb_np(poly)
                        assert hbb[2] > hbb[0] and hbb[3] > hbb[1], "hbb must be valid"
                        
                        if len(obb) == 0:
                            # Weird error in DOTA dataset, skip this instance
                            print("[DATA]", obb, hbb)
                            continue
                    except Exception as e:
                        raise e
                    
                    difficulty = int(bbox_info[9])
                    cls_name = bbox_info[8]
                    label = cls_map[cls_name]

                    gt_difficulty.append(difficulty)
                    gt_bboxes.append(hbb)
                    gt_oboxes.append(obb)
                    gt_labels.append(label)
                    gt_polygons.append(poly)
                    
                    # gt_areas.append((hbb[2] - hbb[0]) * (hbb[3] - hbb[1]))
                    # gt_oareas.append(obb[2] * obb[3])
                                            
            ann = {}
            if gt_bboxes:
                # convert to array then tensor 
                # https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_new.cpp#L262 
                # may be due to memory allocation
                ann['difficulty'] = torch.tensor(np.array(gt_difficulty), dtype=torch.int64)
                ann['image_path'] = img_file
                ann['ann_path'] = ann_file
                ann['bboxes'] = torch.tensor(np.array(gt_bboxes), dtype=torch.float32)
                # ann['area'] = torch.tensor(np.array(gt_areas), dtype=torch.float32)
                
                ann['oboxes'] = torch.tensor(np.array(gt_oboxes), dtype=torch.float32)
                # ann['oarea'] = torch.tensor(np.array(gt_oareas), dtype=torch.float32)
                
                ann['labels'] = torch.tensor(np.array(gt_labels), dtype=torch.int64)
                ann['polygons'] = torch.tensor(np.array(gt_polygons), dtype=torch.float32)
                ann['image_id'] = torch.tensor([image_id], dtype=torch.int64)
                # ann['iscrowd'] = torch.zeros((len(gt_bboxes),), dtype=torch.int64)
                
                image_id += 1
                
            else:
                continue

            anns.append(ann)
        return anns

class BaseXYWHADataset(BaseDataset):
    def load_anns(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains MVTec annotations txt files
        """
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        
        ann_files = glob.glob(ann_folder + '/*.txt')
        # Use images with annotations
        img_files = [ann_file.replace("annfiles", "images").replace(".txt", ".png") for ann_file in ann_files]
        # For det_demo
        if not os.path.isfile(img_files[0]):
            img_files = [ann_file.replace("annfiles", "images").replace(".txt", ".jpg") for ann_file in ann_files]
            assert img_files, "No image files found"
            
        anns = []
        image_id = 0
        for img_file, ann_file in tqdm(zip(img_files, ann_files), desc="Loading annotations", total=len(img_files)):
            gt_difficulty = []
            
            gt_bboxes = []
            gt_oboxes = []
            gt_polygons = []
            
            gt_labels = []
            
            if os.path.getsize(ann_file) == 0:
                continue

            with open(ann_file, "r") as f:
                s = f.readlines()
                for si in s:
                    bbox_info = si.split()
                    
                    obb = np.array(bbox_info[:5], dtype=np.float32)
                    try:
                        poly = box_ops.obb2poly_np(obb)
                        hbb = box_ops.poly2hbb_np(poly)
                        assert hbb[2] > hbb[0] and hbb[3] > hbb[1], "hbb must be valid"
                        
                        if len(obb) == 0:
                            # Weird error in DOTA dataset, skip this instance
                            print("[DATA]", obb, hbb)
                            continue
                        
                    except Exception as e:
                        raise e
                    
                    cls_name = bbox_info[5]
                    difficulty = int(bbox_info[6])
                    label = cls_map[cls_name]

                    gt_difficulty.append(difficulty)
                    gt_bboxes.append(hbb)
                    gt_oboxes.append(obb)
                    gt_labels.append(label)
                    gt_polygons.append(poly)
                    
            ann = {}
            if gt_bboxes:
                # convert to array then tensor 
                # https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_new.cpp#L262 
                # may be due to memory allocation
                ann['difficulty'] = torch.tensor(np.array(gt_difficulty), dtype=torch.int64)
                ann['image_path'] = img_file
                ann['ann_path'] = ann_file
                ann['bboxes'] = torch.tensor(np.array(gt_bboxes), dtype=torch.float32)
                
                ann['oboxes'] = torch.tensor(np.array(gt_oboxes), dtype=torch.float32)
                
                ann['labels'] = torch.tensor(np.array(gt_labels), dtype=torch.int64)
                ann['polygons'] = torch.tensor(np.array(gt_polygons), dtype=torch.float32)
                ann['image_id'] = torch.tensor([image_id], dtype=torch.int64)
                
                image_id += 1
                
            else:
                continue

            anns.append(ann)
        return anns
    
def collate_fn(batch):
    return tuple(zip(*batch))