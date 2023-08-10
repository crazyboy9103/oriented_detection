from datasets import DotaDataModule, MVTecDataModule

mvtec = MVTecDataModule(
    "oc", 
    "xyxy", 
    "/workspace/datasets/mvtec.pth", 
    "/datasets/mvtec_screws", 
    {"batch_size": 1, "num_workers": 4, "shuffle": True, "pin_memory": True}, 
    {"batch_size": 1, "num_workers": 4, "shuffle": False, "pin_memory": True}
)
mvtec.setup("fit")
mvtec.setup("test")

dota = DotaDataModule(
    "oc",
    "xyxy",
    "/workspace/datasets/dota_512.pth",
    "/datasets/split_ss_dota_512",
    {"batch_size": 1, "num_workers": 4, "shuffle": True, "pin_memory": True},
    {"batch_size": 1, "num_workers": 4, "shuffle": False, "pin_memory": True}
)
dota.setup("fit")
dota.setup("test")