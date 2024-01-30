from typing import  Literal, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .base import BaseXYWHADataset, collate_fn

class JMCDataset(BaseXYWHADataset):
    CLASSES = ('background', 'front_retractor', 'back_retractor',
               'front_adjuster', 'back_adjuster', 'front_latch', 'back_latch')

    PALETTE = [(255, 255, 255), (255, 192, 0), (255, 0, 0),
               (0, 32, 96), (0, 176, 80), (112, 48, 160), (0, 112, 192)]
    
    IMAGE_MEAN = (66.033 / 255, 66.816 / 255, 64.153 / 255)
    IMAGE_STD = (47.830 / 255, 45.117 / 255, 44.939 / 255)
    
    def __init__(
        self, 
        save_dir: str = "./jmc.pth",
        data_path: str = "/datasets/jmc",
        split: Literal["train", "test"]="train",
    ):
        super(JMCDataset, self).__init__(save_dir, data_path, split)
    
class JMCDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        save_dir="/workspace/datasets/jmc.pth",
        data_path="/datasets/jmc",
        train_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=True, pin_memory=True), 
        test_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=False, pin_memory=True),
    ):
        super(JMCDataModule, self).__init__()
        self.save_dir = save_dir
        self.data_path = data_path
        
        self.train_loader_kwargs = train_loader_kwargs
        self.test_loader_kwargs = test_loader_kwargs

    def setup(self, stage: Literal["fit", "test"] = "fit") -> None:
        self.train_dataset = JMCDataset(
            save_dir = self.save_dir,
            data_path = self.data_path,
            split = "train",
        )
        self.test_dataset = JMCDataset(
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
