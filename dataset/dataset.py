import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class RPSDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocessing=None):
        self.samples = []
        self.class_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.transform = transform
        self.preprocessing = preprocessing if preprocessing is not None else (lambda img: img)
        for label in self.class_map.keys():
            class_folder = os.path.join(root_dir, label)
            if not os.path.exists(class_folder):
                continue
            for filename in os.listdir(class_folder):
                if filename.lower().endswith((".jpg", ".png")):
                    path = os.path.join(class_folder, filename)
                    img = cv2.imread(path)
                    proc_img = self.preprocessing(img)
                    if proc_img is not None:
                        self.samples.append((path, self.class_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        proc_img = self.preprocessing(img)
        if proc_img is None:
            proc_img = np.zeros((1, 50, 50), dtype=np.uint8)
        img_pil = Image.fromarray(proc_img[0])
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.tensor(proc_img, dtype=torch.float32) / 255.0
        return img_tensor, label
