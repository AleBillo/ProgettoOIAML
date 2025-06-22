import os
import cv2
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Augmentation and Transforms
class Augmentations:
    @staticmethod
    def get_train_transforms():
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(contrast=0.2, brightness=0.2),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    @staticmethod
    def get_test_transforms():
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

# Preprocessing
def preprocess(img, target_size=(50, 50)):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = np.expand_dims(gray, axis=0)
    return gray

# Dataset
class RPSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.class_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.transform = transform
        skipped = 0

        for label in self.class_map.keys():
            class_folder = os.path.join(root_dir, label)
            if not os.path.exists(class_folder):
                continue
            for filename in os.listdir(class_folder):
                if filename.lower().endswith((".jpg", ".png")):
                    path = os.path.join(class_folder, filename)
                    img = cv2.imread(path)
                    proc_img = preprocess(img)
                    if proc_img is not None:
                        self.samples.append((path, self.class_map[label]))
                    else:
                        skipped += 1
        if skipped > 0:
            print(f"Skipped {skipped} images due to failed preprocessing.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        proc_img = preprocess(img)
        if proc_img is None:
            proc_img = np.zeros((1, 50, 50), dtype=np.uint8)
        img_pil = Image.fromarray(proc_img[0])
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.tensor(proc_img, dtype=torch.float32) / 255.0
        return img_tensor, label

# MixUp Utility
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Model
class CNN(nn.Module):
    def __init__(self, input_size=50, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Trainer
class Trainer:
    def __init__(self, model, train_loader, test_loader=None, device=None, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.model.to(self.device)

    def train(self, num_epochs=30, patience=5):
        best_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                imgs, targets_a, targets_b, lam = mixup_data(imgs, labels)
                imgs, targets_a, targets_b = map(torch.autograd.Variable, (imgs, targets_a, targets_b))

                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            if self.test_loader:
                acc = self.evaluate()
                if acc > best_acc:
                    best_acc = acc
                    epochs_no_improve = 0
                    self.save_model("best_model.pth")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("Early stopping triggered.")
                        break
            self.scheduler.step()

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")

# Main Entry
def main():
    train_dir = "dataset/misto/train"
    test_dir = "dataset/misto/test"

    train_dataset = RPSDataset(
        root_dir=train_dir,
        transform=Augmentations.get_train_transforms()
    )
    test_dataset = RPSDataset(
        root_dir=test_dir,
        transform=Augmentations.get_test_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNN(input_size=50, num_classes=3)
    trainer = Trainer(model, train_loader, test_loader)
    trainer.train(num_epochs=30)

if __name__ == "__main__":
    main()