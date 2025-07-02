import torchvision.transforms as T

def get_augmentations(name="default"):
    mapping = {
            "default": {
                "train": T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5]),
                    T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
                    ]),
                "test": T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5])
                    ])
                },
            "simple": {
                "train": T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5])
                    ]),
                "test": T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5])
                    ])
                }
            }

    if name in mapping:
        return mapping[name]
    raise ValueError(f"Unknown augmentation pipeline: {name}")
