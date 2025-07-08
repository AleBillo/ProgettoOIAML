from .trainer import Trainer
from .utils import get_optimizer, get_loss_function, get_scheduler

def get_trainer(**kwargs):
    return Trainer(**kwargs)
