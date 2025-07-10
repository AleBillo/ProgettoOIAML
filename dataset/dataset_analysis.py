import matplotlib.pyplot as plt
import numpy as np
import os

class DatasetAnalysis:
    def __init__(self, dataset, output_dir="analysis"):
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def class_distribution(self, class_map):
        counts = {v: 0 for v in class_map.values()}
        for _, label in self.dataset:
            counts[label] += 1

        labels = [k for k in sorted(counts.keys())]
        values = [counts[k] for k in labels]

        plt.bar(labels, values)
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        output_path = os.path.join(self.output_dir, "class_distribution.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Class distribution plot saved to {output_path}")

    def random_samples_preview(self, class_map, num_samples=5):
        import random
        id_to_class = {v: k for k, v in class_map.items()}

        indices = random.sample(range(len(self.dataset)), num_samples)
        fig, axs = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
        if num_samples == 1:
            axs = [axs]
        for i, idx in enumerate(indices):
            img_tensor, label = self.dataset[idx]
            np_img = (img_tensor.numpy() * 255).astype(np.uint8)
            if np_img.shape[0] == 1:
                np_img = np_img[0]
            else:
                np_img = np.transpose(np_img, (1, 2, 0))
            axs[i].imshow(np_img, cmap='gray')
            axs[i].set_title(f"Label: {label} ({id_to_class[label]})")
            axs[i].axis("off")
        output_path = os.path.join(self.output_dir, "random_samples_preview.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Random samples preview saved to {output_path}")
