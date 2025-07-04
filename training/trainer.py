import os
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from logger.tensorboard_logger import TensorBoardLogger

class Trainer:
    def __init__(
            self,
            model,
            train_dataset,
            test_dataset=None,
            device=None,
            lr=0.001,
            patience=5,
            weight_path="export/weights/best_model.pth",
            batch_size=32,
            resume_from_checkpoint=False,
            checkpoint_path="export/weights/checkpoint.pth",
            tb_logger=None,
            optimizer=None,
            criterion=None,
            scheduler=None
            ):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset is not None else None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = scheduler if scheduler is not None else optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.model.to(self.device)
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.patience = patience
        self.early_stop_counter = 0
        self.weight_path = weight_path
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.tb_logger = tb_logger if tb_logger is not None else TensorBoardLogger()

        if resume_from_checkpoint and os.path.exists(self.checkpoint_path):
            self._load_checkpoint()

        sample_input, _ = train_dataset[0]
        sample_input = sample_input.unsqueeze(0).to(self.device)
        self.tb_logger.log_model_graph(self.model, sample_input)

    def train(self, num_epochs=30):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = total_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total

            test_loss, test_acc = None, None
            if self.test_loader:
                test_loss, test_acc = self.evaluate()

            if self.test_loader and test_acc is not None and test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.weight_path)
                self.early_stop_counter = 0
                print(f"[Epoch {epoch+1}] New best model saved -> {self.weight_path}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            self.tb_logger.log_metrics(epoch, train_loss, test_loss, train_acc, test_acc)
            self._save_checkpoint(epoch, train_loss, train_acc, test_loss, test_acc)
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        self.model.load_state_dict(self.best_model_wts)

    def evaluate(self):
        if self.test_loader is None:
            return None, None
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(self.test_loader)
        acc = 100.0 * correct / total
        self.tb_logger.log_confusion_matrix(0, all_preds, all_labels, list(range(3)))
        return avg_loss, acc

    def _save_checkpoint(self, epoch, train_loss, train_acc, test_loss, test_acc):
        checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
                }
        torch.save(checkpoint, self.checkpoint_path)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_acc = checkpoint["best_acc"]
        print(f"Resumed training from checkpoint at epoch {checkpoint['epoch']+1}")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")
