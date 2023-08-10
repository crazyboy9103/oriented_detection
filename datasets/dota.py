from typing import  Literal, Callable, Optional, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .base import BaseDataset

class DotaDataset(BaseDataset):
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255), (0, 128, 128)]
    
    def __init__(
        self, 
        save_dir: str = "./dota.pth",
        data_path: str = "/datasets/split_ss_dota_512",
        angle_version="oc", 
        hbb_version: Literal["xyxy", "xywh"]="xyxy",
        split: Literal["train", "test"]="train",
        transform: Optional[Callable]=None
    ):
        """
        Args:
            angle_version: angle version of the dataset, one of ["oc", "le90", "le135"]. Currently only "oc" is supported.
        
        """
        super(DotaDataset, self).__init__(save_dir, data_path, angle_version, hbb_version, split, transform)
    
class DotaDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        angle_version="oc", 
        hbb_version="xyxy", 
        save_dir="/workspace/datasets/dota_512.pth",
        data_path="/datasets/split_ss_dota_512",
        train_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=True, pin_memory=True), 
        test_loader_kwargs: Dict[str, Any] = dict(batch_size=1, num_workers=4, shuffle=False, pin_memory=True),
    ):
        super(DotaDataModule, self).__init__()
        self.angle_version = angle_version
        self.hbb_version = hbb_version
        self.save_dir = save_dir
        self.data_path = data_path
        
        self.train_loader_kwargs = train_loader_kwargs
        self.test_loader_kwargs = test_loader_kwargs
        
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Literal["fit", "test"] = "fit") -> None:
        if stage == "fit":
            self.train_dataset = DotaDataset(
                save_dir = self.save_dir,
                data_path = self.data_path,
                angle_version = self.angle_version, 
                hbb_version = self.hbb_version,
                split = "train",
                transform = None
            )

        if stage == "test":
            self.test_dataset = DotaDataset(
                save_dir = self.save_dir,
                data_path = self.data_path,
                angle_version = self.angle_version, 
                hbb_version = self.hbb_version,
                split = "test",
                transform = None
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_loader_kwargs)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.test_loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.test_loader_kwargs)
