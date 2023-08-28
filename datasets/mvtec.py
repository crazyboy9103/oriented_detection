from typing import Literal, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .base import BaseDataset, collate_fn

class MVTecDataset(BaseDataset):
    # TODO merge fine-grained classes
    CLASSES = ('background', 'nut', 'wood_screw', 'lag_wood_screw', 'bolt', 
               'black_oxide_screw', 'shiny_screw', 'short_wood_screw', 'long_lag_screw', 
               'large_nut', 'nut2', 'nut1', 'machine_screw', 
               'short_machine_screw')

    PALETTE = [(255, 255, 255), (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0)]

    IMAGE_MEAN = (211.35 / 255, 166.559 / 255, 97.271 / 255)
    IMAGE_STD = (43.849 / 255, 40.172 / 255, 30.459 / 255)


    def __init__(
        self, 
        save_dir: str = "/workspace/datasets/mvtec.pth",
        data_path: str = "/datasets/split_ss_mvtec",
        split: Literal["train", "test"]="train",
    ):
        super(MVTecDataset, self).__init__(save_dir, data_path, split)
    
class MVTecDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        save_dir="/workspace/datasets/mvtec.pth",
        data_path="/datasets/split_ss_mvtec",
        train_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=True, pin_memory=True), 
        test_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=False, pin_memory=True),
    ):
        super(MVTecDataModule, self).__init__()
        self.save_dir = save_dir
        self.data_path = data_path
        
        self.train_loader_kwargs = train_loader_kwargs
        self.test_loader_kwargs = test_loader_kwargs
        
    def setup(self, stage: Literal["fit", "test"] = "fit") -> None:
        self.train_dataset = MVTecDataset(
            save_dir = self.save_dir,
            data_path = self.data_path,
            split = "train",
        )
        self.test_dataset = MVTecDataset(
            save_dir = self.save_dir,
            data_path = self.data_path,
            split = "test",
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, collate_fn=collate_fn, **self.train_loader_kwargs)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, collate_fn=collate_fn, **self.test_loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, collate_fn=collate_fn,**self.test_loader_kwargs)
