from typing import  Literal, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .base import BaseDataset, collate_fn

class DetDemoDataset(BaseDataset):
    CLASSES = ('background', 'spray', 'foam', 'settingspray', 'sanitizer',
               'wet_tissue', 'cylindrical', 'handcream')

    PALETTE = [(255, 255, 255), (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255)]
    
    IMAGE_MEAN = (123.675 / 255, 116.28 / 255, 103.53 / 255)
    IMAGE_STD = (58.395 / 255, 57.12 / 255, 57.375 / 255)
    
    def __init__(
        self, 
        save_dir: str = "./detdemo.pth",
        data_path: str = "/datasets/detdemo",
        split: Literal["train", "test"]="train",
    ):
        super(DetDemoDataset, self).__init__(save_dir, data_path, split)
    
class DetDemoDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        save_dir="/workspace/datasets/detdemo.pth",
        data_path="/datasets/detdemo",
        train_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=True, pin_memory=True), 
        test_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=False, pin_memory=True),
    ):
        super(DetDemoDataModule, self).__init__()
        self.save_dir = save_dir
        self.data_path = data_path
        
        self.train_loader_kwargs = train_loader_kwargs
        self.test_loader_kwargs = test_loader_kwargs

    def setup(self, stage: Literal["fit", "test"] = "fit") -> None:
        self.train_dataset = DetDemoDataset(
            save_dir = self.save_dir,
            data_path = self.data_path,
            split = "train",
        )
        self.test_dataset = DetDemoDataset(
            save_dir = self.save_dir,
            data_path = self.data_path,
            split = "test",
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, collate_fn=collate_fn, **self.train_loader_kwargs)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, collate_fn=collate_fn, **self.test_loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, collate_fn=collate_fn, **self.test_loader_kwargs)
