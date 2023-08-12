from datasets import DotaDataModule, MVTecDataModule

train_loader_kwargs = dict(batch_size=1, num_workers=4, shuffle=True, pin_memory=True)
test_loader_kwargs = dict(batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

mvtec = MVTecDataModule(
    "oc", 
    "xyxy", 
    "/workspace/datasets/mvtec.pth", 
    "/datasets/split_ss_mvtec", 
    train_loader_kwargs, 
    test_loader_kwargs
)
mvtec.setup()

dota_256 = DotaDataModule(
    "oc",
    "xyxy",
    "/workspace/datasets/dota_256.pth",
    "/datasets/split_ss_dota_256",
    train_loader_kwargs,
    test_loader_kwargs
)
dota_256.setup()

dota_512 = DotaDataModule(
    "oc",
    "xyxy",
    "/workspace/datasets/dota_512.pth",
    "/datasets/split_ss_dota_512",
    train_loader_kwargs,
    test_loader_kwargs
)
dota_512.setup()

dota_800 = DotaDataModule(
    "oc",
    "xyxy",
    "/workspace/datasets/dota_800.pth",
    "/datasets/split_ss_dota_800",
    train_loader_kwargs,
    test_loader_kwargs
)
dota_800.setup()