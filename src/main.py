import torch
from torch.utils.data import DataLoader
from augmentations import Augmentations
from dataset import RPSDataset
from model import CNN
from trainer import Trainer
import os

def main():
    train_dir = "dataset/pulito/train"
    test_dir = "dataset/pulito/test"

    os.makedirs("export/weights", exist_ok=True)

    train_dataset = RPSDataset(root_dir=train_dir, transform=Augmentations.get_train_transforms())
    test_dataset = RPSDataset(root_dir=test_dir, transform=Augmentations.get_test_transforms())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNN(input_size=50, num_classes=3)
    trainer = Trainer(model, train_loader, test_loader)
    trainer.train(num_epochs=30)
    
    trainer.save_model("export/weights/model.pth")

if __name__ == "__main__":
    main()
