import os
import torch
import copy
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader=None,
        device=None,
        lr=0.001,
        patience=5,
        weight_path="export/weights/best_model.pth",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.model.to(self.device)
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.patience = patience
        self.early_stop_counter = 0
        self.weight_path = weight_path
        os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)

    def train(self, num_epochs=30):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            if self.test_loader:
                acc = self.evaluate()
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.best_model_wts, self.weight_path)
                    self.early_stop_counter = 0
                    print(f"New best model saved to {self.weight_path}")
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print("Early stopping triggered.")
                        break

            self.scheduler.step()

        # Load best weights at end
        self.model.load_state_dict(self.best_model_wts)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100.0 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
        return acc

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")