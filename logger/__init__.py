from .tensorboard_logger import TensorBoardLogger

def get_logger(logging_config):
    name = logging_config.get("name", "tensorboard")
    if name == "tensorboard":
        log_dir = logging_config.get("log_dir", "runs/rps_experiment")
        return TensorBoardLogger(log_dir=log_dir)
    raise ValueError(f"Unknown logging backend: {name}")
