import os
import json
import jsonschema
from dataset.dataset import RPSDataset
from dataset.augmentations import get_augmentations
from training.trainer import Trainer
from nets import get_model
from preprocess import get_preprocessor
from training import get_optimizer, get_loss_function, get_scheduler
from dataset.dataset_analysis import DatasetAnalysis
from logger import get_logger


def load_and_validate_config(config_path, schema_path):
    with open(schema_path, "r") as f_schema:
        schema = json.load(f_schema)
    with open(config_path, "r") as f_config:
        config = json.load(f_config)
    jsonschema.validate(instance=config, schema=schema)
    return config


def main():
    config = load_and_validate_config("config/config.json", "config/config_schema.json")

    aug = get_augmentations(config.get("augmentation", "default"))
    train_transform = aug["train"]
    test_transform = aug["test"]
    preprocessing_method = config.get("preprocessing", "greyscale")
    preprocessing_fn = get_preprocessor(preprocessing_method)

    train_dataset = RPSDataset(
            root_dir=config["paths"]["train_dir"],
            transform=train_transform,
            preprocessing=preprocessing_fn
            )
    test_dataset = RPSDataset(
            root_dir=config["paths"]["test_dir"],
            transform=test_transform,
            preprocessing=preprocessing_fn
            )

    analyzer = DatasetAnalysis(train_dataset, output_dir="analysis")
    analyzer.class_distribution(train_dataset.class_map)
    analyzer.random_samples_preview(train_dataset.class_map, num_samples=5)

    model = get_model(config["model"], input_size=50, num_classes=3)
    optimizer = get_optimizer(model, config["optimizer"])
    criterion = get_loss_function(config["loss"])
    scheduler = get_scheduler(optimizer, config["scheduler"])
    tb_logger = get_logger(config["logging"])

    trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            lr=config["optimizer"].get("lr", 0.001),
            patience=config["training"]["patience"],
            weight_path=os.path.join(config["paths"]["weight_dir"], "best_model.pth"),
            batch_size=config["training"]["batch_size"],
            resume_from_checkpoint=config["training"]["resume_from_checkpoint"],
            checkpoint_path=config["training"]["checkpoint_path"],
            tb_logger=tb_logger,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler
            )

    trainer.train(num_epochs=config["training"]["num_epochs"])
    trainer.save_model(os.path.join(config["paths"]["weight_dir"], "model.pth"))

    print("Training complete.")

if __name__ == "__main__":
    main()
