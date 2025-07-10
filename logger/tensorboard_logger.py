from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class TensorBoardLogger:
    def __init__(self, log_dir="runs/rps_experiment"):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metrics_resume = {}

    def log_model_graph(self, model, input_tensor):
        self.writer.add_graph(model, input_tensor)

    def log_metrics(self, epoch, train_loss=None, test_loss=None, train_acc=None, test_acc=None):
        if train_loss is not None:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
        if test_loss is not None:
            self.writer.add_scalar("Loss/Test", test_loss, epoch)
        if train_acc is not None:
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
        if test_acc is not None:
            self.writer.add_scalar("Accuracy/Test", test_acc, epoch)

        self.metrics_resume[epoch] = {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                }

    def log_histograms(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"{name}.grad", param.grad, epoch)

    def log_confusion_matrix(self, epoch, preds, labels, class_names):
        cm = confusion_matrix(labels, preds)
        figure = self._plot_confusion_matrix(cm, class_names)
        self.writer.add_figure("Confusion Matrix", figure, epoch)

    def _plot_confusion_matrix(self, cm, class_names):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color=color)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        fig.tight_layout()
        return fig

    def save_resume(self, analysis_folder):
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)

        resume_filepath = os.path.join(analysis_folder, "training_resume.txt")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        config_path = os.path.join("config", "config.json")
        try:
            with open(config_path, "r") as cfgf:
                options = json.load(cfgf)
            options_str = json.dumps(options, indent=2)
        except Exception as e:
            options_str = f"Could not load config/config.json: {e}"

        with open(resume_filepath, "w") as f:
            f.write(f"Training summary generated at: {now}\n")
            f.write("Options/configuration:\n")
            f.write(options_str)
            f.write("\n\n")
            for epoch in sorted(self.metrics_resume.keys()):
                metrics = self.metrics_resume[epoch]
                f.write(f"epoch {epoch}\n")
                if metrics.get("train_loss") is not None:
                    f.write(f"- train loss: {metrics.get('train_loss')}\n")
                if metrics.get("test_loss") is not None:
                    f.write(f"- test loss: {metrics.get('test_loss')}\n")
                if metrics.get("train_acc") is not None:
                    f.write(f"- train accuracy: {metrics.get('train_acc')}\n")
                if metrics.get("test_acc") is not None:
                    f.write(f"- test accuracy: {metrics.get('test_acc')}\n")
                f.write("\n")
        print(f"Training resume saved to {resume_filepath}")

    def close(self):
        self.writer.close()
